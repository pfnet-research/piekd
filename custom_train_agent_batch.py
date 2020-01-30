from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA

from collections import deque
import logging

import numpy as np

#from chainerrl.experiments.evaluator import Evaluator
from custom_evalutor import Evaluator, MultipleAgentEvaluator
from chainerrl.experiments.evaluator import save_agent
from chainerrl.misc.makedirs import makedirs

def softmax(x):
    """Compute the softmax in a numerically stable way."""""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def parallel_train_agent_batch(start_weighted_size, all_agents, env, steps, outdir, log_interval=None,
                      max_episode_len=None, eval_interval=None,
                      step_offset=0, evaluator=None, before_evaluator=None, successful_score=None,
                      step_hooks=[], return_window_size=100, logger=None, step_callback=None, schedule_args={}):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        steps (int): Number of total time steps for training.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See chainerrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)

    # TODO: set a buffer for recent returns
    n_agents = len(all_agents)
    select_probs = np.array([1 / n_agents] * n_agents)
    agent_ids = [id(agent) for agent in all_agents]
    assert len(np.unique(agent_ids)) == n_agents

    all_recent_returns = [deque(maxlen=return_window_size) for i in range(len(all_agents))]
    all_eval_returns = [0 for i in range(len(all_agents))]
    num_envs = env.num_envs
    assert num_envs == 1

    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype='i')
    episode_len = np.zeros(num_envs, dtype='i')

    # o_0, r_0
    obss = env.reset()
    rs = np.zeros(num_envs, dtype='f')

    t = step_offset

    # TODO: initialize an agent index for the first episode (initially we use uniform distribution)
    perf_metric = schedule_args['perf_metric']
    selected_agent_idx = np.random.randint(low=0, high=len(all_agents))
    logger.info('selected_idx: {}'.format(selected_agent_idx))
    try:
        while True:
            # TODO: take the actins from the selected agent
            agent = all_agents[selected_agent_idx]
            # a_t
            actions = agent.batch_act_and_train(obss)
            # HACK: to set the other agents' batch_last_obs as []
            for idx in range(n_agents):
                if idx != selected_agent_idx:
                    all_agents[idx].batch_last_obs = [None] * num_envs

            # o_{t+1}, r_{t+1}
            obss_next, rs, dones, infos = env.step(actions)
            obss = obss_next
            episode_r += rs
            episode_len += 1

            # Compute mask for done and reset
            if max_episode_len is None:
                resets = np.zeros(num_envs, dtype=bool)
            else:
                resets = (episode_len == max_episode_len)
            resets = np.logical_or(
                resets, [info.get('needs_reset', False) for info in infos])

            # Agent observes the consequences
            for idx, agent_in_list in enumerate(all_agents):
                # Only the selected agent will add replay buffer. The others just update.
                agent_in_list.batch_observe_and_train(obss, rs, dones, resets)
                logger.debug('agent_{}: t = {}, selected_idx = {}'.format(idx, agent_in_list.t, selected_agent_idx))
            after_ts = [agent_in_list.t for agent_in_list in all_agents]
            assert len(np.unique(after_ts)) == 1

            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)

            if before_evaluator:
                before_max_mean, before_all_means = before_evaluator.evaluate_if_necessary(t=t + 1, episodes=np.sum(episode_idx))

            if step_callback is not None:
                new_select_probs = step_callback(t, all_recent_returns, all_eval_returns, np.any(end))
                if new_select_probs is not None and schedule_args['select_prob_update'] == 'interval':
                    select_probs = new_select_probs

            # For episodes that ends, do the following:
            #   1. increment the episode count
            #   2. record the return
            #   3. clear the record of rewards
            #   4. clear the record of the number of steps
            #   5. reset the env to start a new episode
            # 3-5 are skipped when training is already finished.
            episode_idx += end

            # TODO: append to the selected returns
            all_recent_returns[selected_agent_idx].extend(episode_r[end])

            for _ in range(num_envs):
                t += 1
                for hook in step_hooks:
                    hook(env, agent, t)

            if (log_interval is not None
                    and t >= log_interval
                    and t % log_interval < num_envs):
                logger.info(
                        'outdir:{} agent_idx: {} step:{} episode:{} last_R: {} average_R:{}'.format(  # NOQA
                        outdir,
                        selected_agent_idx,
                        t,
                        np.sum(episode_idx),
                        all_recent_returns[selected_agent_idx][-1] if all_recent_returns[selected_agent_idx] else np.nan,
                        np.mean(all_recent_returns[selected_agent_idx]) if all_recent_returns[selected_agent_idx] else np.nan,
                    ))
                logger.info('statistics: {}'.format(agent.get_statistics()))
            if evaluator:
                max_mean, all_means = evaluator.evaluate_if_necessary(t=t, episodes=np.sum(episode_idx))
                all_eval_returns = all_means
                if max_mean:
                    if (successful_score is not None and
                            evaluator.max_score >= successful_score):
                        break

            if t >= steps:
                break

            # Start new episodes if needed
            episode_r[end] = 0
            episode_len[end] = 0
            obss = env.reset(not_end)

            # TODO: update the selected agent index
            if np.any(end):
                logger.debug('selected_agent.last_obs: {}'.format(agent.batch_last_obs))
                #TODO: weighted selection, uniform, or eps-greedy
                if t < start_weighted_size or schedule_args['select_algo'] == 'uniform':
                    # prevent not yet eval
                    if t < start_weighted_size:
                        all_mean_returns = '(warm up)'
                    else:
                        if perf_metric == 'train':
                            all_mean_returns = np.asarray([np.mean(recent_returns) for recent_returns in all_recent_returns])
                        else:
                            all_mean_returns = np.asarray(all_eval_returns)
                    logger.info('t: {} (new agent idx) selection prob.: (uniform), mean returns: {}'.format(t, all_mean_returns))
                    selected_agent_idx = np.random.randint(low=0, high=len(all_agents))
                else:
                    if perf_metric == 'train':
                        all_mean_returns = np.asarray([np.mean(recent_returns) for recent_returns in all_recent_returns])
                    else:
                        all_mean_returns = np.asarray(all_eval_returns)

                    # determine sample or greedy using eps or algo
                    if schedule_args['select_algo'] == 'eps-greedy':
                        eps = schedule_args['eps_schedule'].value(t)
                        use_sample = (np.random.random() < eps)
                        logger.info('t: {}, eps: {}, sample: {}, mean returns: {}'.format(t, eps, use_sample, all_mean_returns))
                        selected_agent_idx = np.random.choice(n_agents, 1)[0] if use_sample else np.argmax(all_mean_returns)
                    else:
                        temp = 1.0
                        if 'select_prob_temp' in schedule_args:
                            temp = schedule_args['select_prob_temp']
                            logger.info('use temperature: {}'.format(schedule_args['select_prob_temp']))
                        if schedule_args['select_prob_update'] == 'episode':
                            select_probs = softmax(all_mean_returns / temp)
                        else:
                            logger.info('Use periodically update')
                        logger.info('t: {} (new agent idx) selection prob.: {}, mean returns: {}'.format(t, select_probs, all_mean_returns))
                        selected_agent_idx = np.random.choice(n_agents, 1, p=select_probs)[0]

                logger.info('t: {} selected_idx: {}'.format(t, selected_agent_idx))

            # TODO: Call step callbacks

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        for idx, agent in enumerate(all_agents):
            save_agent(agent, t, outdir, logger, suffix='_except_{}'.format(idx))
        env.close()
        if evaluator:
            evaluator.env.close()
        raise
    else:
        # Save the final model
        for idx, agent in enumerate(all_agents):
            save_agent(agent, t, outdir, logger, suffix='_finish_{}'.format(idx))

def parallel_train_agent_batch_with_evaluation(start_weighted_size, all_agents,
                                      env,
                                      steps,
                                      eval_n_steps,
                                      eval_n_episodes,
                                      eval_interval,
                                      outdir,
                                      max_episode_len=None,
                                      step_offset=0,
                                      eval_max_episode_len=None,
                                      return_window_size=100,
                                      eval_env=None,
                                      log_interval=None,
                                      successful_score=None,
                                      step_hooks=[],
                                      save_best_so_far_agent=True,
                                      logger=None,
                                      step_callback=None,
                                      schedule_args={},
                                      eval_before_distill=False,
                                      ):
    logger = logger or logging.getLogger(__name__)

    makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = MultipleAgentEvaluator(all_agents=all_agents,
                          n_steps=eval_n_steps,
                          n_episodes=eval_n_episodes,
                          eval_interval=eval_interval, outdir=outdir,
                          max_episode_len=eval_max_episode_len,
                          env=eval_env,
                          step_offset=step_offset,
                          save_best_so_far_agent=save_best_so_far_agent,
                          logger=logger,
                          )

    if eval_before_distill:
        before_evaluator = MultipleAgentEvaluator(all_agents=all_agents,
                          n_steps=eval_n_steps,
                          n_episodes=eval_n_episodes,
                          eval_interval=eval_interval, outdir=outdir,
                          max_episode_len=eval_max_episode_len,
                          env=eval_env,
                          step_offset=step_offset,
                          save_best_so_far_agent=save_best_so_far_agent,
                          logger=logger,
                          suffix='-before-distillation'
                          )
    else:
        before_evaluator = None

    parallel_train_agent_batch(start_weighted_size,
        all_agents, env, steps, outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        eval_interval=eval_interval,
        evaluator=evaluator, before_evaluator=before_evaluator,
        successful_score=successful_score,
        return_window_size=return_window_size,
        log_interval=log_interval,
        step_hooks=step_hooks,
        logger=logger,
        step_callback=step_callback, schedule_args=schedule_args)

def train_agent_batch(agent, env, steps, outdir, log_interval=None,
                      max_episode_len=None, eval_interval=None,
                      step_offset=0, evaluator=None, successful_score=None,
                      step_hooks=[], return_window_size=100, logger=None, step_callback=None):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        env: Environment to train the agent against.
        steps (int): Number of total time steps for training.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See chainerrl.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)
    recent_returns = deque(maxlen=return_window_size)

    num_envs = env.num_envs
    episode_r = np.zeros(num_envs, dtype=np.float64)
    episode_idx = np.zeros(num_envs, dtype='i')
    episode_len = np.zeros(num_envs, dtype='i')

    # o_0, r_0
    obss = env.reset()
    rs = np.zeros(num_envs, dtype='f')

    t = step_offset
    if hasattr(agent, 't'):
        agent.t = step_offset

    try:
        while True:
            # a_t
            actions = agent.batch_act_and_train(obss)
            # o_{t+1}, r_{t+1}
            obss_next, rs, dones, infos = env.step(actions)
            if step_callback is not None:
                step_callback(locals(), globals())
            obss = obss_next
            episode_r += rs
            episode_len += 1

            # Compute mask for done and reset
            if max_episode_len is None:
                resets = np.zeros(num_envs, dtype=bool)
            else:
                resets = (episode_len == max_episode_len)
            resets = np.logical_or(
                resets, [info.get('needs_reset', False) for info in infos])
            # Agent observes the consequences
            agent.batch_observe_and_train(obss, rs, dones, resets)

            # Make mask. 0 if done/reset, 1 if pass
            end = np.logical_or(resets, dones)
            not_end = np.logical_not(end)

            # For episodes that ends, do the following:
            #   1. increment the episode count
            #   2. record the return
            #   3. clear the record of rewards
            #   4. clear the record of the number of steps
            #   5. reset the env to start a new episode
            # 3-5 are skipped when training is already finished.
            episode_idx += end
            recent_returns.extend(episode_r[end])

            for _ in range(num_envs):
                t += 1
                for hook in step_hooks:
                    hook(env, agent, t)

            if (log_interval is not None
                    and t >= log_interval
                    and t % log_interval < num_envs):
                logger.info(
                    'outdir:{} step:{} episode:{} last_R: {} average_R:{}'.format(  # NOQA
                        outdir,
                        t,
                        np.sum(episode_idx),
                        recent_returns[-1] if recent_returns else np.nan,
                        np.mean(recent_returns) if recent_returns else np.nan,
                    ))
                logger.info('statistics: {}'.format(agent.get_statistics()))
            if evaluator:
                if evaluator.evaluate_if_necessary(
                        t=t, episodes=np.sum(episode_idx)):
                    if (successful_score is not None and
                            evaluator.max_score >= successful_score):
                        break

            if t >= steps:
                break

            # Start new episodes if needed
            episode_r[end] = 0
            episode_len[end] = 0
            obss = env.reset(not_end)

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix='_except')
        env.close()
        if evaluator:
            evaluator.env.close()
        raise
    else:
        # Save the final model
        save_agent(agent, t, outdir, logger, suffix='_finish')


def train_agent_batch_with_evaluation(agent,
                                      env,
                                      steps,
                                      eval_n_steps,
                                      eval_n_episodes,
                                      eval_interval,
                                      outdir,
                                      max_episode_len=None,
                                      step_offset=0,
                                      eval_max_episode_len=None,
                                      return_window_size=100,
                                      eval_env=None,
                                      log_interval=None,
                                      successful_score=None,
                                      step_hooks=[],
                                      save_best_so_far_agent=True,
                                      logger=None,
                                      step_callback=None,
                                      ):
    """Train an agent while regularly evaluating it.

    Args:
        agent: Agent to train.
        env: Environment train the againt against.
        steps (int): Number of total time steps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If set to None, max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See chainerrl.experiments.hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)

    makedirs(outdir, exist_ok=True)

    if eval_env is None:
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    evaluator = Evaluator(agent=agent,
                          n_steps=eval_n_steps,
                          n_episodes=eval_n_episodes,
                          eval_interval=eval_interval, outdir=outdir,
                          max_episode_len=eval_max_episode_len,
                          env=eval_env,
                          step_offset=step_offset,
                          save_best_so_far_agent=save_best_so_far_agent,
                          logger=logger,
                          )

    train_agent_batch(
        agent, env, steps, outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        eval_interval=eval_interval,
        evaluator=evaluator,
        successful_score=successful_score,
        return_window_size=return_window_size,
        log_interval=log_interval,
        step_hooks=step_hooks,
        logger=logger,
        step_callback=step_callback)
