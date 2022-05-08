import random

class Agent:
    def __init__(self, states, actions, epsilon=0.7, alpha=0.7, gamma=0.7, epsilon_decay=0.9):
        self.q = [[0]*actions]*states
        self.actions = [i for i in range(actions)]
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.last_state = None
        self.last_action = None
        self.last_reward = None
    
    def update(self, state, action, reward, new_value):
        current_q = self.get_q(state, action)
        if current_q == 0:
            self.set_q(state, action, reward) 
        else:
            self.set_q(state, action, current_q + self.alpha * (new_value - current_q))
            
    def learn(self, state, action, reward, state2, action2):
        pass
        
    def act(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            state_q = [self.get_q(state, a) for a in self.actions] 
            max_q = max(state_q)
            count = state_q.count(max_q)
            if count > 1:
                max_actions = [a for a in self.actions if state_q[a] == max_q] 
                index = random.choice(max_actions)
            else:
                index = state_q.index(max_q)

            action = self.actions[index]
            self.epsilon = self.epsilon * self.epsilon_decay
            
        return action
    
    def get_q(self, state, action):
        return self.q[state][action]

    def set_q(self, state, action, update_value):
        self.q[state][action] = update_value

class SarsaAgent(Agent):
    
    def learn(self, state, action, reward, state2, action2):
        next_q = self.get_q(state2, action2)
        self.update(state, action, reward, reward + self.gamma * next_q)

class QLearningAgent(Agent):
    
    def learn(self, state, action, reward, state2, _):
        max_q = max([self.get_q(state2, a) for a in self.actions])
        self.update(state, action, reward, reward + self.gamma * max_q)
