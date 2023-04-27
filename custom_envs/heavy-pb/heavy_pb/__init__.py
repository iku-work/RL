from gym.envs.registration import register
register(
    id='driving-v0',
    entry_point='heavy_pb.envs:SimpleDrivingEnv',
    max_episode_steps=100
)

register(
    id='wheel-driving-v0',
    entry_point='heavy_pb.envs:WheelSimpleDrivingEnv',
    max_episode_steps=2000
)

register(
    id='forwarder-v0',
    entry_point='heavy_pb.envs:ForwarderPick',
    max_episode_steps=400
)