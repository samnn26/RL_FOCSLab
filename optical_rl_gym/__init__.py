from gym.envs.registration import register

register(
    id='RMSA-v0',
    entry_point='optical_rl_gym.envs:RMSAEnv',
)

register(
    id='DeepRMSA-v0',
    entry_point='optical_rl_gym.envs:DeepRMSAEnv',
)

register(
    id='RWA-v0',
    entry_point='optical_rl_gym.envs:RWAEnv',
)

register(
    id='RWAFOCS-v1',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV1',
)

register(
    id='RWAFOCS-v2',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV2',
)

register(
    id='RWAFOCS-v3',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV3',
)

register(
    id='FOCSLabRWA-v0',
    entry_point='optical_rl_gym.envs:FOCSLabRWAEnv',
)

register(
    id='QoSConstrainedRA-v0',
    entry_point='optical_rl_gym.envs:QoSConstrainedRA',
)
