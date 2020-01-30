from gym.envs.registration import registry, register, make, spec

register(
    id='SparseMountainCarContinuous-v0',
    entry_point='envs.sparse_continuous_mountain_car:Sparse_Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='SparseHalfCheetah-v2',
    entry_point='envs.half_cheetah:SparseHalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='SparseReacher-v2',
    entry_point='envs.reacher:SparseReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='SparseHopper-v2',
    entry_point='envs.hopper:SparseHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='SparseWalker2d-v2',
    max_episode_steps=1000,
    entry_point='envs.walker2d:SparseWalker2dEnv',
)

register(
    id='SparseAnt-v2',
    entry_point='envs.ant:SparseAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='SparseHumanoid-v2',
    entry_point='envs.humanoid:SparseHumanoidEnv',
    max_episode_steps=1000,
)



