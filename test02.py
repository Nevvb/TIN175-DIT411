import sys
import gym
import random

import numpy as np
import matplotlib.pyplot as plt
from gym import spaces

# --- Specific cartpole q-learning

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')

q = dict()

size_up = 10
actions = env.action_space
action_space = [0, 1] # TODO Fix (generic) action space
print("actions: "+str(actions))

epsilon = 1.0   # Epsilon greedy probability
alpha = 0.2     # Learning rate
gamma = 0.5     # Discount factor

def max_a(state):
    best_reward = -1.0
    best_actions = []
    if state in q:
        for action, reward in q[state].items():
            if reward > best_reward:
                best_actions = [action]
                best_reward = reward
            elif reward == best_reward:
                best_actions.append(action)
        action_chosen = random.randint(0, len(best_actions) - 1)
        # print(f'action chosen: {best_actions[action_chosen]}')
        return best_actions[action_chosen]
    else:
        q[state] = dict()
        for action in action_space:
            q[state][action] = 0.0
        return random.randint(0, len(action_space) - 1)

def prob_a(state):
    if state in q:
        action_probs = {}
        prob_sum = 0
        for action, reward in q[state].items():
            prob = np.exp(reward)
            action_probs[action] = prob
            prob_sum += prob
        random_prob = random.random() * prob_sum
        for action, prob in action_probs.items():
            random_prob -= prob
            if random_prob <= 0:
                return action
        return -1 # something went wrong here
    else:
        q[state] = dict()
        for action in action_space:
            q[state][action] = 0.0
        return random.randint(0, len(action_space) - 1)

def state_from_space(space):
    return str(np.rint(10 * space)[1:4].astype(int))

# for plotting
times = []
avg_times = []
max_times = []
min_times = []

# for stopping
success_counter = 0

# decides how often we should draw
draw_interval = 200

# Running X episodes

X = 20000
for i_episode in range(X):

    # Resetting environment
    observation = env.reset()
    current_state = state_from_space(observation)
    epsilon = np.exp(-i_episode * 100 / X)
    if i_episode % draw_interval == 0: # 999:  
        # epsilon *= 0.5
        print(f'epsilon={epsilon}')
    if i_episode % draw_interval == 0:
        print(f'Episode {i_episode}')

    state_actions_taken = {}
    next_states = {}

    # Running through an episode
    for t in range(1000):
        if i_episode % draw_interval == 0:
            env.render()

        # Select action
        
        # for state in current_state:
        action = max_a(current_state)
        if random.random() < epsilon:
            action = env.action_space.sample()

        # Perform action, observe reward and new state
        observation, reward, done, info = env.step(action)
        new_state = state_from_space(observation)

        if current_state in state_actions_taken:
            state_actions_taken[current_state].append(action)
            next_states[current_state][action] = new_state
        else:
            state_actions_taken[current_state] = [action]
            next_states[current_state] = {}
            next_states[current_state][action] = new_state
        
        # Updating state
        current_state = new_state

        # Aborting episode if done
        if done:
            # print(f'Episode finished after {t + 1} timesteps')
            break

    # Q-learning equation
    reward = ((t + 1) / 200) - 1
    for current_state, actions in state_actions_taken.items():
        for action in actions:
            new_state = next_states[current_state][action]
            if not current_state in q:
                q[current_state] = {}
                for possible_action in action_space:
                    q[current_state][possible_action] = 0.0
            if not new_state in q:
                q[new_state] = {}
                for possible_action in action_space:
                    q[new_state][possible_action] = 0.0
            q[current_state][action] = (1 - alpha) * q[current_state][action] + alpha * ((1 - gamma) * reward + gamma * q[new_state][max_a(new_state)])
            # q[current_state][action] += reward
    last_items = times[i_episode - 99:i_episode]
    last_items.append(t + 1)
    # print(f'last_items: {last_items}, list size: {len(last_items)}')
    times.append(t + 1)
    mean = np.mean(last_items)
    max_ = np.max(last_items)
    min_ = np.min(last_items)
    if i_episode % draw_interval == 0:
        print(f'Episode finished after {t + 1} timesteps')
        print(f'Mean score: {mean}')
    if mean >= 195.0:
        print(f'Success! After {i_episode} episodes')
        print(f'Mean score: {mean}')
        success_counter += 1
        if success_counter >= 20:
            break
    else:
        success_counter = 0
    avg_times.append(mean)
    max_times.append(max_)
    min_times.append(min_)
plt.plot(times)
plt.plot(avg_times)
plt.plot(max_times)
plt.plot(min_times)
plt.show()
sys.exit()
