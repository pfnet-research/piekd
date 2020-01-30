"""A training script of Soft Actor-Critic on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1812.05905 as much
as possible.
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import functools
import logging
import os
import sys
import copy

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
from chainer import cuda

import gym
import gym.wrappers
import numpy as np
import cupy as cp
import logging

import chainerrl
from chainerrl import experiments
from chainerrl import misc
from chainerrl import replay_buffer

from distill import train_bc_batch, reset_model_params, copy_model_params
import custom_train_agent_batch

from schedules import LinearSchedule, ConstantSchedule
from sparse_wrapper import SparseRewardWrapper

from chainerrl.replay_buffer import batch_experiences

import roboschool
import envs

def softmax(x):
    """Compute the softmax in a numerically stable way."""""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def concat_obs_and_action(obs, action):
    """Concat observation and action to feed the critic."""
    return F.concat((obs, action), axis=-1)

def make_mlp(n_layers, h, activ=F.relu, initW=chainer.initializers.GlorotUniform()):
    layers = []
    for i in range(n_layers):
        layers.append(L.Linear(None, h, initialW=initW))
        layers.append(activ)
    return chainer.Sequential(*layers)

def make_q_func_with_optimizer(n_layers, h):
    winit = chainer.initializers.GlorotUniform()
    q_func = chainer.Sequential(
        concat_obs_and_action,
        make_mlp(n_layers, h),
        L.Linear(None, 1, initialW=winit),
    )
    q_func_optimizer = optimizers.Adam(3e-4).setup(q_func)
    return q_func, q_func_optimizer

def make_pi_with_optimizer(n_layers, h, action_size, policy_output_scale, policy_lr):
    winit_policy_output = chainer.initializers.GlorotUniform(policy_output_scale)

    def squashed_diagonal_gaussian_head(x):
        assert x.shape[-1] == action_size * 2
        mean, log_scale = F.split_axis(x, 2, axis=1)
        log_scale = F.clip(log_scale, -20, 2)
        var = F.exp(log_scale * 2)
        return chainerrl.distribution.SquashedGaussianDistribution(
            mean, var=var)

    policy = chainer.Sequential(
        make_mlp(n_layers, h),
        L.Linear(None, action_size * 2, initialW=winit_policy_output),
        squashed_diagonal_gaussian_head,
    )
    policy_optimizer = optimizers.Adam(policy_lr).setup(policy)

    return policy, policy_optimizer

def make_model(obs_size, action_size, n_layers, h,  policy_output_scale, lr):
    pi, pi_optimizer = make_pi_with_optimizer(n_layers, h, action_size, policy_output_scale, lr)
    q_func1, q_func1_optimizer = make_q_func_with_optimizer(n_layers, h)
    q_func2, q_func2_optimizer = make_q_func_with_optimizer(n_layers, h)
    return {'pi': pi, 'q1': q_func1, 'q2': q_func2}, {'pi': pi_optimizer, 'q1': q_func1_optimizer, 'q2': q_func2_optimizer}

def parse_eps_schedule(eps_schedule_str):
    algo, extra_args = eps_schedule_str.split(':')
    if algo == 'linear':
        steps, init_p, final_p = map(float, extra_args.split(','))
        steps = int(steps)
        return LinearSchedule(steps, initial_p=init_p, final_p=final_p)
    elif algo == 'const':
        val = extra_args.split(',')[0]
        return ConstantSchedule(float(val))
    else:
        raise NotImplemented()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--env', type=str, default='Hopper-v2', required=True,
                        help='OpenAI Gym MuJoCo env to perform algorithm on.')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of envs run in parallel.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--load', type=str, default='',
                        help='Directory to load agent from.')
    parser.add_argument('--steps', type=int, default=10 ** 6,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--eval-n-runs', type=int, default=20,
                        help='Number of episodes run for each evaluation.')
    parser.add_argument('--eval-interval', type=int, default=10000,
                        help='Interval in timesteps between evaluations.')
    parser.add_argument('--replay-start-size', type=int, default=10000,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Minibatch size')
    parser.add_argument('--render', action='store_true',
                        help='Render env states in a GUI window.')
    parser.add_argument('--demo', action='store_true',
                        help='Just run evaluation, not training.')
    parser.add_argument('--monitor', action='store_true',
                        help='Wrap env with gym.wrappers.Monitor.')
    parser.add_argument('--log-interval', type=int, default=1000,
                        help='Interval in timesteps between outputting log'
                             ' messages during training')
    parser.add_argument('--logger-level', type=int, default=logging.INFO,
                        help='Level of the root logger.')
    parser.add_argument('--policy-output-scale', type=float, default=1.,
                        help='Weight initialization scale of polity output.')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode.')

    # Environment related
    parser.add_argument('--sparse-level', type=int, default=0)
    parser.add_argument('--noise-scale', type=float, default=0.0)

    parser.add_argument('--all-h', type=str, default='256,256,256')
    parser.add_argument('--all-d', type=str, default='2,2,2')
    parser.add_argument('--all-lr', type=str, default='3e-4,3e-4,3e-4')

    parser.add_argument('--cpo-temp', type=float, default=1.0)
    parser.add_argument('--cpo-train-sample', type=str, default='all', choices=['sample', 'all'])
    parser.add_argument('--cpo-num-train-batch', type=int, default=20)
    parser.add_argument('--cpo-select-algo', type=str, default='uniform', choices=['uniform', 'softmax', 'eps-greedy'])
    parser.add_argument('--cpo-select-prob-update', type=str, default='episode', choices=['interval', 'episode'])
    parser.add_argument('--cpo-eps-schedule', type=str, default='linear:500000,1.0,0.1', help='linear:steps,init,final or const:val')
    parser.add_argument('--cpo-recent-returns-interval', type=int, default=5)
    parser.add_argument('--cpo-distill-interval', type=int, default=5000)
    parser.add_argument('--cpo-distill-batch-size', type=int, default=256)
    parser.add_argument('--cpo-distill-epochs', type=int, default=5)
    parser.add_argument('--cpo-distill-lr', type=float, default=1e-3)
    parser.add_argument('--cpo-distill-bc-coef', type=float, default=0.8)
    parser.add_argument('--cpo-distill-type', type=str, default='bc', choices=['bc', 'rlbc'])
    parser.add_argument('--cpo-distill-pi-only', action='store_true', default=False)
    parser.add_argument('--cpo-distill-q-only', action='store_true', default=False)
    parser.add_argument('--cpo-random-distill', action='store_true', default=False)
    parser.add_argument('--cpo-bad-distill', action='store_true', default=False)
    parser.add_argument('--cpo-distill-reset-returns', action='store_true', default=False)
    parser.add_argument('--cpo-distill-perf-metric', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--cpo-distill-reset-model', action='store_true', default=False)
    parser.add_argument('--cpo-distill-schedule', type=str, choices=['fix', 'ada'], default='fix')
    parser.add_argument('--cpo-distill-ada-threshold', type=float, default=0.8)
    parser.add_argument('--cpo-test-extra-update', action='store_true', default=False)
    parser.add_argument('--cpo-eval-before-distill', action='store_true', default=False)
    parser.add_argument('--cpo-use-hardcopy', action='store_true', default=False)
    parser.add_argument('--cpo-mutual-learning', action='store_true', default=False)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    if args.debug:
        chainer.set_debug(True)

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)

    consoleHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler("{}/log.txt".format(args.outdir))
    logging.basicConfig(level=args.logger_level, handlers=[consoleHandler, fileHandler])
    print('Output files are saved in {}'.format(args.outdir))


    #fileHandler.setFormatter(logFormatter)
    #logger.addHandler(fileHandler)

    #consoleHandler.setFormatter(logFormatter)
    #logger.addHandler(consoleHandler)

    misc.set_random_seed(args.seed, gpus=(args.gpu,))
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(process_idx, test):
        if args.env.startswith('DM'):
            import dm_wrapper
            domain, task = args.env.split('-')[1:]
            env = dm_wrapper.make(domain_name=domain, task_name=task)
            timestep_limit = env.dmcenv._step_limit
        else:
            env = gym.make(args.env)

            # Unwrap TimiLimit wrapper
            assert isinstance(env, gym.wrappers.TimeLimit)
            env = env.env
            timestep_limit = env.spec.tags.get(
                'wrapper_config.TimeLimit.max_episode_steps')

            # Wrapped with FlattenDict
            if isinstance(env, gym.GoalEnv):
                keys = env.observation_space.spaces.keys()
                logger.info('GoalEnv: {}'.format(keys))
                env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = chainerrl.wrappers.CastObservationToFloat32(env)
        # Normalize action space to [-1, 1]^n
        env = chainerrl.wrappers.NormalizeActionSpace(env)
        # Sparsify the reward signal if needed
        if args.sparse_level > 0:
            env = SparseRewardWrapper(env, args.sparse_level, timestep_limit)
        if args.noise_scale > 0:
            from noise_wrapper import NoiseWrapper
            env = NoiseWrapper(env, scale=args.noise_scale)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [functools.partial(make_env, idx, test)
             for idx, env in enumerate(range(args.num_envs))])

    sample_env = make_env(process_idx=0, test=False)
    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    obs_space = sample_env.observation_space
    obs_size = np.asarray(obs_space.shape).prod()
    action_space = sample_env.action_space
    print('Observation space:', obs_space)
    print('Action space:', action_space)

    action_size = action_space.low.size

    all_h = [int(h) for h in args.all_h.split(',')]
    all_d = [int(d) for d in args.all_d.split(',')]
    all_lr = [float(lr) for lr in args.all_lr.split(',')]
    assert len(all_h) == len(all_d) and len(all_d) == len(all_lr)

    rbuf = replay_buffer.ReplayBuffer(10 ** 6)

    def make_agent(h, d, lr):
        funcs, optimizers = make_model(obs_size, action_size, d, h, args.policy_output_scale, lr)
        policy, policy_optimizer = funcs['pi'], optimizers['pi']
        q_func1, q_func1_optimizer = funcs['q1'], optimizers['q1']
        q_func2, q_func2_optimizer = funcs['q2'], optimizers['q2']

        # Draw the computational graph and save it in the output directory.
        fake_obs = chainer.Variable(
            policy.xp.zeros_like(obs_space.low, dtype=np.float32)[None],
            name='observation')
        fake_action = chainer.Variable(
            policy.xp.zeros_like(action_space.low, dtype=np.float32)[None],
            name='action')
        chainerrl.misc.draw_computational_graph(
            [policy(fake_obs)], os.path.join(args.outdir, 'policy'))
        chainerrl.misc.draw_computational_graph(
            [q_func1(fake_obs, fake_action)], os.path.join(args.outdir, 'q_func1'))
        chainerrl.misc.draw_computational_graph(
            [q_func2(fake_obs, fake_action)], os.path.join(args.outdir, 'q_func2'))

        def burnin_action_func():
            """Select random actions until model is updated one or more times."""
            return np.random.uniform(
                action_space.low, action_space.high).astype(np.float32)

        # Hyperparameters in http://arxiv.org/abs/1802.09477
        agent = chainerrl.agents.SoftActorCritic(
            policy,
            q_func1,
            q_func2,
            policy_optimizer,
            q_func1_optimizer,
            q_func2_optimizer,
            rbuf,
            gamma=0.99,
            replay_start_size=args.replay_start_size,
            gpu=args.gpu,
            minibatch_size=args.batch_size,
            burnin_action_func=burnin_action_func,
            entropy_target=-action_size,
            temperature_optimizer=chainer.optimizers.Adam(3e-4),
            use_mutual_learning=args.cpo_mutual_learning
        )

        return agent

    env = make_batch_env(test=False)
    eval_env = make_batch_env(test=True)
    all_agents = [make_agent(h, d, lr) for h, d, lr in zip(all_h, all_d, all_lr)]
    if args.cpo_mutual_learning:
        for i in range(len(all_agents)):
            all_agents[i].set_mutual_learning(all_agents, i)

    def distill_to_agent(teacher_agent, student_agent, history_obses, history_obses_acs, fix_batch_num, num_train_batch):
        if not args.cpo_distill_pi_only:
            qf1_distill_losses = train_bc_batch(target_model=teacher_agent.q_func1, model=student_agent.q_func1, loss_fn=F.mean_squared_error,
                                        train_inputs=history_obses_acs,
                                        batch_size=args.cpo_distill_batch_size, lr=args.cpo_distill_lr,
                                        n_epochs=args.cpo_distill_epochs,
                                        predict_fn=lambda m, x: m(x[:, :obs_size], x[:, obs_size:]), fix_batch_num=fix_batch_num, num_batch=num_train_batch)
            logger.info('Qf1 distill min loss: {}'.format(np.min(qf1_distill_losses)))

            qf2_distill_losses = train_bc_batch(target_model=teacher_agent.q_func2, model=student_agent.q_func2, loss_fn=F.mean_squared_error,
                                        train_inputs=history_obses_acs,
                                        batch_size=args.cpo_distill_batch_size, lr=args.cpo_distill_lr,
                                        n_epochs=args.cpo_distill_epochs,
                                        predict_fn=lambda m, x: m(x[:, :obs_size], x[:, obs_size:]), fix_batch_num=fix_batch_num, num_batch=num_train_batch)
            logger.info('Qf2 distill min losses: {}'.format(np.min(qf2_distill_losses)))
        else:
            qf1_distill_losses = qf2_distill_losses = None

        if not args.cpo_distill_q_only:
            def rlbc_loss(inputs, pred, targ):
                bc_loss = F.mean(targ.kl(pred))

                batch_state = inputs

                action_distrib = pred
                actions, log_prob = pred.sample_with_log_prob()
                q1 = teacher_agent.q_func1(batch_state, actions)
                q2 = teacher_agent.q_func2(batch_state, actions)
                q = F.minimum(q1, q2)

                entropy_term = student_agent.temperature * log_prob[..., None]
                assert q.shape == entropy_term.shape
                rl_loss = F.mean(entropy_term - q)

                return (args.cpo_distill_bc_coef) * bc_loss + (1.0 - args.cpo_distill_bc_coef) * rl_loss

            if args.cpo_distill_type == 'rlbc':
                logger.info('Use RL+BC')
                pi_distill_losses = train_bc_batch(target_model=teacher_agent.policy, model=student_agent.policy,
                                        loss_fn=rlbc_loss,
                                        train_inputs=history_obses,
                                        batch_size=args.cpo_distill_batch_size, lr=args.cpo_distill_lr,
                                        n_epochs=args.cpo_distill_epochs, with_inputs=True, fix_batch_num=fix_batch_num, num_batch=num_train_batch)
            elif args.cpo_distill_type == 'bc':
                logger.info('Use BC')
                pi_distill_losses = train_bc_batch(target_model=teacher_agent.policy, model=student_agent.policy,
                                        loss_fn=lambda pred, targ: F.mean(targ.kl(pred)),
                                        train_inputs=history_obses,
                                        batch_size=args.cpo_distill_batch_size, lr=args.cpo_distill_lr,
                                        n_epochs=args.cpo_distill_epochs, fix_batch_num=fix_batch_num, num_batch=num_train_batch)
            logger.info('Pi distill min losses: {}'.format(np.min(pi_distill_losses)))
        else:
            pi_distill_losses = None

        return qf1_distill_losses, qf2_distill_losses, pi_distill_losses

    def extra_update(agent, all_experiences):
        for epoch in range(args.cpo_distill_epochs):
            n_samples = len(all_experiences)
            indices = np.asarray(range(n_samples))
            np.random.shuffle(indices)
            for start_idx in (range(0, n_samples, args.cpo_distill_batch_size)):
                batch_idx = indices[start_idx:start_idx + args.cpo_distill_batch_size].astype(np.int32)
                experiences = [all_experiences[idx] for idx in batch_idx]
                batch = batch_experiences(experiences, agent.xp, agent.phi, agent.gamma)
                agent.update_policy_and_temperature(batch)

    def distill_callback(t, all_recent_returns, all_eval_returns, end):
        if t > args.replay_start_size:
            if args.cpo_distill_perf_metric == 'eval':
                all_mean_returns = np.asarray(all_eval_returns)
            else:
                all_mean_returns = np.asarray([np.mean(recent_returns) for recent_returns in all_recent_returns])
            temp = args.cpo_temp
            all_weights = softmax(all_mean_returns / temp)

            if args.cpo_distill_schedule == 'fix':
                require_distill = t % args.cpo_distill_interval == 0
            else:
                if end:
                    logger.info('Prob.: {}'.format(all_weights))
                require_distill = np.max(all_weights) >= args.cpo_distill_ada_threshold

            if require_distill:
                if args.cpo_test_extra_update:
                    #assert len(all_agents) == 1
                    all_experiences = [e for e in rbuf.memory]
                    #agent = all_agents[0]
                    for agent in all_agents:
                        extra_update(agent, all_experiences)
                        logger.info('Did extra update')
                else:
                    history_obses = np.asarray([e[0]['state'] for e in rbuf.memory])
                    history_acs = np.asarray([e[0]['action'] for e in rbuf.memory])
                    history_obses_acs = np.concatenate([history_obses, history_acs], axis=1)

                    if args.cpo_random_distill:
                        best_agent_idx = np.random.randint(len(all_mean_returns))
                        logger.info('Random distill')
                    elif args.cpo_bad_distill:
                        best_agent_idx = np.argmin(all_mean_returns)
                        logger.info('Bad distill')
                    else:
                        logger.info('Best distill')
                        best_agent_idx = np.argmax(all_mean_returns)
                    best_agent = all_agents[best_agent_idx]

                    for idx, other_agent in enumerate(all_agents):
                        if idx != best_agent_idx:
                            if args.cpo_use_hardcopy:
                                copy_model_params(best_agent.q_func1, other_agent.q_func1)
                                copy_model_params(best_agent.q_func2, other_agent.q_func2)
                                copy_model_params(best_agent.policy, other_agent.policy)
                                logger.info('Copy done')
                            else:
                                if args.cpo_distill_reset_model:
                                    logger.info('Reset model')
                                    reset_model_params(other_agent.q_func1)
                                    reset_model_params(other_agent.q_func2)
                                    reset_model_params(other_agent.policy)
                                logger.info('Distill to {} from {}'.format(idx, best_agent_idx))
                                qf1_losses, qf2_losses, pi_losses = distill_to_agent(best_agent, other_agent, history_obses, history_obses_acs,
                                        fix_batch_num=True if args.cpo_train_sample == 'sample' else False, num_train_batch=args.cpo_num_train_batch)
                                other_agent.sync_target_network()

                            if args.cpo_distill_reset_returns:
                                logger.info('Reset returns')
                                all_recent_returns[idx] = copy.copy(all_recent_returns[best_agent_idx])
                return all_weights
        return None

    custom_train_agent_batch.parallel_train_agent_batch_with_evaluation(
        start_weighted_size=args.replay_start_size,
        all_agents=all_agents,
        env=env,
        eval_env=eval_env,
        outdir=args.outdir,
        steps=args.steps,
        eval_n_steps=None,
        eval_n_episodes=args.eval_n_runs,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        max_episode_len=timestep_limit,
        save_best_so_far_agent=True,
        schedule_args={'select_prob_temp': args.cpo_temp,
                        'select_prob_update': args.cpo_select_prob_update,
                        'select_algo': args.cpo_select_algo,
                        'eps_schedule': parse_eps_schedule(args.cpo_eps_schedule),
                        'perf_metric': args.cpo_distill_perf_metric},
        step_callback=distill_callback,
        return_window_size=args.cpo_recent_returns_interval,
        eval_before_distill=args.cpo_eval_before_distill
    )

    env.close()
    eval_env.close()

if __name__ == '__main__':
    main()
