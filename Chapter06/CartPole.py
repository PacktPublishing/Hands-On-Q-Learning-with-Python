
# coding: utf-8

# In[72]:

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


gamma = 0.95
alpha = 0.5
learning_rate_adam = 0.01
epsilon = 0.999
epsilon_decay = 0.99


class DQN:

    def __init__(self, observation_space, action_space):
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.memory = []
        self.batch_size = 32

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=learning_rate_adam))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        q = self.model.predict(state)
        return np.argmax(q[0])

    def update(self):
        mem_sample = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in mem_sample:
            update_value = reward
            if not done:
                update_value = self.alpha * (reward + self.gamma * np.max(self.model.predict(next_state)[0]))
            q = self.model.predict(state)
            q[0][action] = update_value
            self.model.fit(state, q, verbose=0)
        self.epsilon *= epsilon_decay

    def memory_update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

def cartpole():
    env = gym.make("CartPole-v1")
    observation_space, action_space = env.observation_space.shape[0], env.action_space.n
    
    dqn = DQN(observation_space, action_space)
    epoch = 0
    while True:
        score = 0
        epoch += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        while True:
            score += 1
            action = dqn.choose_action(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -reward
            next_state = np.reshape(next_state, [1, observation_space])
            dqn.memory_update(state, action, reward, next_state, done)
            state = next_state
            if done:
                print ("Epoch: " + str(epoch) + " Score: " + str(score))
                break
            dqn.update()


if __name__ == "__main__":
    cartpole()


# In[ ]:




# In[ ]:



