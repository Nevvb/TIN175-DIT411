import gym
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import datetime

print("Gym:", gym.__version__)

#env_name = "Taxi-v2"
env_name = "BreakoutDeterministic-v4"
#env_name = "CartPole-v0"
env = gym.make(env_name).env
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
state = env.reset()


def toGrayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downSample(img):
    return img[::2, ::2]

def preProcess(img):
    return toGrayscale(downSample(img))
#q_table = np.zeros([env.observation_space.n, env.action_space.n])
#q_table = [state, env.action_space.n]
#print(env.observation_space.shape[0]*env.observation_space.shape[1])
q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])
#print(state)

#Hyperparameters
alpha = 0.7     #Learning rate
gamma = 1     #Accumulated discount reward
epsilon = 0.3   #Rate of exploration

#For plotting metrics
# for plotting
times = []
avg_times = []
max_times = []
min_times = []
all_rewards = []

num_episodes = np.power(10, 5)
#num_episodes = 20
i = 0

for ep in range(num_episodes):
    
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

        observation, reward, done, info = env.step(action)

        next_state = preProcess(observation)
        #print(next_state)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma *next_max)
        q_table[state, action] = new_value

        state = next_state

        epochs += 1
        #env.render()
        tot_reward += reward
        
        
        if done:
            print('Episode {}' .format(i), 'finished with {} reward' .format(tot_reward))
            all_rewards.append(tot_reward)
            #epsilon = np.maximum(0.1, epsilon * 0.9)
            #print("all rewards {}".format(all_rewards))
            i += 1
            last_items = times[ep - 99:ep]
            last_items.append(epochs)
            times.append(epochs)
            mean = np.mean(last_items)
            max_ = np.max(last_items)
            min_ = np.min(last_items)

            avg_times.append(mean)
            max_times.append(max_)
            min_times.append(min_)
            break

print(q_table) 
with open('test.csv', 'w') as fp:
    for row in q_table:
        for item in row:
            fp.write("%s" % item)
            fp.write(",")
        fp.write("\n")
    fp.close    

with open('test.txt', 'w') as f:
    for item in all_rewards:
        f.write("%s\n" % item)
    f.close

print("Runtime:", datetime.timedelta(seconds = time.process_time()))
plt.plot(all_rewards)
plt.plot(times)
plt.plot(avg_times)
plt.plot(max_times)
plt.plot(min_times)
plt.plot([0, len(times)], [195, 195])
plt.show()


#print(all_rewards)
