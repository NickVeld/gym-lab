#!/usr/bin/env python
import gym

from pathenv.agent import BaseAgent

episodes_number = 1000
max_steps = 200

if __name__ == '__main__':
    env = gym.make('PathFindingByPixelWithDistanceMapEnv-v1')
    env._configure()

    agent = BaseAgent(input_shape=env.observation_space.shape, number_of_actions=env.action_space.n)

    for episode_i in xrange(1, episodes_number + 1):

        observation = env.reset()
        agent.new_episode(env.finish)

        reward, done = 0, False

        for step_i in range(max_steps):
            action = agent.act(observation)
            next_observation, reward, done, _ = env.step(action)
            agent.update_memory(observation, action, reward, next_observation, done)
            observation = next_observation
            if done:
                break

        agent.train_on_memory()
