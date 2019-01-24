#set up gym

import gym
env = gym.make('Taxi-v2')

#create a game loop and take random actions

observation = env.reset()
reward = 0
while reward != 20:
    observation, reward, done, info = env.step(env.action_space.sample())
env.render()

    
