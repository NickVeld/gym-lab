import gym.envs.registration

gym.envs.registration.register('PathFindingByPixelWithDistanceMapEnv-v1',
                               entry_point='pathenv.environ:PathFindingByPixelWithDistanceMapEnv')