import gym
import numpy as np
import random
import tensorflow as tf

env = gym.make('Taxi-v2')

tf.reset_default_graph()
inputs = tf.placeholder(shape=[1, env.observation_space.n], dtype=tf.float32)
weights = tf.Variable(tf.random_uniform([env.observation_space.n,env.action_space.n], 0, 0.01))
q_out = tf.matmul(inputs, weights)
predict = tf.argmax(q_out,1)

next_q = tf.placeholder(shape=[1,env.action_space.n],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_q - q_out))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
loss_update = trainer.minimize(loss)
init = tf.global_variables_initializer()

total_epochs = 0
total_rewards = 0

gamma = 0.7
epsilon = 0.2
epsilon_decay = .99
episodes = 100

with tf.Session() as sess:
    sess.run(init)
    for episode in range(episodes):
        state = env.reset()
        rewards_this_episode = 0
        epochs = 0

        done = False
        
        while not done:
            action, q = sess.run([predict,q_out], feed_dict={inputs:np.identity(env.observation_space.n)[state:state + 1]})
            
            if np.random.rand(1) < epsilon:
                action[0] = env.action_space.sample()
                
            next_state, reward, done, info = env.step(action[0])
            
            curr_q = sess.run(q_out, feed_dict = {inputs:np.identity(env.observation_space.n)[next_state:next_state+1]})
            max_next_q = np.max(curr_q)
            target_q = q
            target_q[0, action[0]] = reward + gamma * max_next_q
            
            info, new_weights = sess.run([loss_update, weights], feed_dict={inputs:np.identity(env.observation_space.n)[state:state+1], next_q:target_q})
            rewards_this_episode += reward
            state = next_state
            epochs += 1
            
        epsilon = epsilon * epsilon_decay
        total_epochs += epochs
        total_rewards += rewards_this_episode
        
print ("Success rate: " + str(total_rewards/episodes))
