import gym
import random
import numpy as np

print("Gym:", gym.__version__)

#env_name = "Taxi-v2"
env_name = "Breakout-v0"
env = gym.make(env_name).env
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

#q_table = np.zeros([env.observation_space.n, env.action_space.n])
q_table = np.zeros([300, env.action_space.n])

#Hyperparameters
alpha = 0.1     #Learning rate
gamma = 0.6     #Accumulated discount reward
epsilon = 0.1   #Rate of exploration

#For plotting metrics
all_epochs = []


num_episodes = 2000
all_rewards = np.zeros([num_episodes])

for ep in range(num_episodes):
    i = 0
    state = env.reset()

    epochs, penalties, reward, tot_reward = 0, 0, 0, 0

    done = False

    while not done:
        #Explore
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        #Exploit
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma *next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state

        epochs += 1
        #env.render()
        tot_reward += reward
        
        
        if done:
            print('Episode finished after {} timesteps' .format(epochs+1))
            all_rewards[i] = tot_reward
            print("all rewards {}".format(all_rewards))
            break
    i += 1

print('success')