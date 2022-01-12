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
    id='RWAFOCS-v21',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV2_1',
)
register(
    id='RWAFOCS-v22',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV2_2',
)
register(
    id='RWAFOCS-v23',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV2_3',
)
register(
    id='RWAFOCS-v24',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV2_4',
)
register(
    id='RWAFOCS-v25',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV2_5',
)

register(
    id='RWAFOCS-v3',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV3',
)

register(
    id='RWAFOCS-v4',
    entry_point='optical_rl_gym.envs:RWAEnvFOCSV4',
)

register(
    id='FOCSLabRWA-v0',
    entry_point='optical_rl_gym.envs:FOCSLabRWAEnv',
)

register(
    id='QoSConstrainedRA-v0',
    entry_point='optical_rl_gym.envs:QoSConstrainedRA',
)
