import gym
import random
import matplotlib.pyplot as plt
import numpy as np
import csv

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
q_table = np.zeros([env.observation_space.shape[0]*env.observation_space.shape[1], env.action_space.n])
#print(state)

#Hyperparameters
alpha = 0.1     #Learning rate
gamma = 1     #Accumulated discount reward
epsilon = 0.3   #Rate of exploration

#For plotting metrics
all_epochs = []
all_rewards = []

num_episodes = np.power(10, 2)
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
            #print('Episode {}' .format(i), 'finished with {} reward' .format(tot_reward))
            all_rewards.append(tot_reward)
            #epsilon = np.maximum(0.1, epsilon * 0.9)
            #print("all rewards {}".format(all_rewards))
            i += 1
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

plt.plot(all_rewards)
plt.show()
#print(all_rewards)
