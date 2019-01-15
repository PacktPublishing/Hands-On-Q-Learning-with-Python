#set up gym

import gym
env = gym.make('Taxi-v2')
env.reset()

#create a game loop and take random actions

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) 

    
    
    
    
    
    
