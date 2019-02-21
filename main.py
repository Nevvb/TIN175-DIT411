import gym
import random
import numpy as np

print("Gym:", gym.__version__)

env_name = "Taxi-v2"
env = gym.make(env_name).env
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

#Hyperparameters
alpha = 0.1     #Learning rate
gamma = 0.6     #Accumulated discount reward
epsilon = 0.1   #Rate of exploration

#For plotting metrics
all_epochs = []
all_penalties = []

num_episodes = 20

for ep in range(num_episodes):
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0

    done = False

    for  i in range(100):
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
        env.render()
        print(state)
        
        
        if done:
            print('Episode finished after {} timesteps' .format(i+1))
            break

print('success')