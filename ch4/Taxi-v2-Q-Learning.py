import gym
import numpy as np

#create environment
env = gym.make("Taxi-v2")

#initialize Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

gamma = 0.1
alpha = 0.1
epsilon = 0.1
total_epochs = 0
episodes = 100

for episode in range(episodes):
    epochs = 0
    reward = 0
    state = env.reset()
    
    while reward != 20:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        next_state, reward, done, info = env.step(action)
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * \
                            np.max(Q[next_state]) - Q[state, action])
        state = next_state 
        epochs += 1
    total_epochs += epochs
    
print("Average timesteps taken: {}".format(total_epochs/episodes))
