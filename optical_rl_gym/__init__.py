from gym.envs.registration import register

register(
    id='RMSA-v0',
    entry_point='optical_rl_gym.envs:RMSAEnv',
)

register(
    id='Test2-v0',
    entry_point='optical_rl_gym.envs:Test2Env',
)

register(
    id='Test-v0',
    entry_point='optical_rl_gym.envs:TestEnv',
)

register(
    id='RWA-v0',
    entry_point='optical_rl_gym.envs:RWAEnv',
)
register(
    id='OPT-v0',
    entry_point='optical_rl_gym.envs:OpticalNetworkEnv',
)
register(
    id='Test3-v0',
    entry_point='optical_rl_gym.envs:Test3Env',
)
