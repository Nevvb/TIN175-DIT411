import sys
import gym
import random
import math
from gym import logger

import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

# --- Specific cartpole q-learning

logger.set_level(logger.ERROR)
env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('Breakout-v0')

# Q table
q = dict()

# Setting up parameters
size_up = 10
actions = env.action_space
action_space = [0, 1] # TODO Fix (generic) action space
max_timer = 200

epsilon = 1.0   # Epsilon greedy probability
alpha = 0.8     # Learning rate
gamma = 0.4     # Discount factor

# Finding the best action to take, given a certain state
def max_a(state):
    best_reward = None
    best_actions = []
    # If state has been visited before, choose an action with best reward:
    if state in q:
        for action, reward in q[state].items():
            if best_reward == None or reward > best_reward:
                best_actions = [action]
                best_reward = reward
            elif reward == best_reward:
                best_actions.append(action)
        action_chosen = random.randint(0, len(best_actions) - 1)
        return best_actions[action_chosen]
    # If state has never been visited, chose an ation by random:
    else:
        q[state] = dict()
        for action in action_space:
            q[state][action] = 0.0
        return random.randint(0, len(action_space) - 1)

# Converts space to workable state
def state_from_space(space):
    inverted = False
    if space[2] < 0:
        inverted = True
        space *= -1
    int_array = np.rint(10 * space)[1:4].astype(int)
    return str(int_array), inverted

# For plotting
times = []
avg_times = []
max_times = []
min_times = []
mega_time = 600
success_threshold = 195

# For stopping when algorithm has succeeded
success_counter = 0

# Decides how often we should draw
draw_interval = 200

# Running X episodes
X = 20000
theta_threshold_radians = 12 * 2 * math.pi / 360
for i_episode in range(X):

    # Resetting environment
    observation = env.reset()
    current_state, inverted = state_from_space(observation)
    epsilon = np.exp(-i_episode * 100 / X)
    if i_episode % draw_interval == 0:
        print(f'epsilon={epsilon}')
        print(f'Episode {i_episode}')

    state_actions_taken = {}
    next_states = {}
    iterations_to_ignore = 0

    # Running through an episode
    for t in range(mega_time):
        # Only render at a certain interval
        if i_episode % draw_interval == 0:
            env.render()

        # Select action
        action = max_a(current_state)
        if random.random() < epsilon:
            action = env.action_space.sample()
        
        actual_action = action
        if inverted:
            actual_action = 1 - action

        # Perform action, observe reward and new state
        observation, reward, done, info = env.step(actual_action)
        new_state, inverted = state_from_space(observation)

        if not new_state in q:
            q[new_state] = {}
            for possible_action in action_space:
                q[new_state][possible_action] = 0.0

        # Performing Q-equation
        if done and t < max_timer - 1:
            if i_episode % draw_interval == 0:
                print('BAD')
            reward = -1000
            q[current_state][action] = (1 - alpha) * q[current_state][action] + alpha * (reward + gamma * q[new_state][max_a(new_state)])
        if not done:
            q[current_state][action] = (1 - alpha) * q[current_state][action] + alpha * (reward + gamma * q[new_state][max_a(new_state)])

        # Updating state
        current_state = new_state

        # Aborting episode if done
        if done:
            break

    # Logging results after an episode, and printing episode-specific results
    times.append(t + 1 - iterations_to_ignore)
    last_items = times[i_episode - 99:i_episode+1]

    mean = np.mean(last_items)
    max_ = np.max(last_items)
    min_ = np.min(last_items)
    if i_episode % draw_interval == 0:
        print(f'Episode finished after {t + 1} timesteps')
        print(f'Mean score: {mean}')
        print(f'q size: {len(q)}')
    if mean >= success_threshold:
        print(f'Success! After {i_episode} episodes')
        print(f'Mean score: {mean}')
        print(f'q size: {len(q)}')
        success_counter += 1
        if success_counter >= 5:
            break
    else:
        success_counter = 0
    avg_times.append(mean)
    max_times.append(max_)
    min_times.append(min_)

# Plotting results
plt.plot(avg_times)
plt.plot(max_times)
plt.plot(min_times)
plt.plot([0, len(times)], [success_threshold, success_threshold])
plt.legend(['Mean score', 'Max score', 'Min score', 'Success threshold'])
plt.title('Moving average, max and min scores over last 100 runs')
plt.show()
