import gym
import random
import matplotlib.pyplot as plt
import numpy as np
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

# - Decompression of image by converting to greyscale and shrinking size
def toGrayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

def downSample(img):
    return img[::2, ::2]

def preProcess(img):
    return toGrayscale(downSample(img))

# - Creating Q table
q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])

# - Hyperparameters
alpha = 0.7     #Learning rate
gamma = 1     #Accumulated discount reward
epsilon = 0.3   #Rate of exploration

# - For plotting metrics
times = []
avg_times = []
max_times = []
min_times = []
all_rewards = []

# - How many episodes to run
num_episodes = np.power(10, 4)
#num_episodes = 20

# - Episode counter
i = 0

for ep in range(num_episodes):
    
    # - New episode resets
    state = env.reset()
    epochs, penalties, reward, tot_reward = 0, 0, 0, 0
    done = False

    # - One episode:
    while not done:
        #Explore
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        #Exploit
        else:
            action = np.argmax(q_table[state])

        # - Taking one step
        observation, reward, done, info = env.step(action)

        # - Reducing state complexity (to have less states to travel)
        next_state = preProcess(observation)

        # - Helping variables for the Q equation
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # - Calculating new q-value and saving it to table
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma *next_max)
        q_table[state, action] = new_value

        # - Updating current state
        state = next_state

        # - Only render every 20th episode, for speed
        if i % 20 == 0:
            env.render()

        epochs += 1
        tot_reward += reward
        
        # - When episode is done (mostly when all lives are out)
        if done:
            # - Increasing episode
            print('Episode {}' .format(i), 'finished with {} reward' .format(tot_reward))
            i += 1

            # - Logging for result and plots later
            all_rewards.append(tot_reward)
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

# - Printing and saving results for debugging
print(q_table) 

# Showing results in a plot
print("Runtime:", datetime.timedelta(seconds = time.process_time()))
plt.plot(all_rewards)
plt.plot(times)
plt.plot(avg_times)
plt.plot(max_times)
plt.plot(min_times)
plt.plot([0, len(times)], [195, 195])
plt.show()


#print(all_rewards)
