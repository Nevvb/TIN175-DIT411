import sys
import gym
import random

import numpy as np
from gym import spaces

# --- Specific cartpole q-learning

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')

q = dict()

size_up = 10
actions = env.action_space
print("actions: "+str(actions))

epsilon = 0.1   # Epsilon greedy probability
alpha = 0.2     # Learning rate
gamma = 0.5     # Discount factor

def max_a(state):
    best_reward = 0.0
    best_actions = []
    if state in q:
        for action, reward in q[state].items():
            if reward > best_reward:
                best_actions = [action]
                best_reward = reward
            elif reward == best_reward:
                best_actions.append(action)
        if best_reward != 0.0:
            print(f'actions: {best_actions}, reward: {best_reward}')
        action_chosen = random.randint(0, len(best_actions) - 1)
        # print(f'action chosen: {best_actions[action_chosen]}')
        return best_actions[action_chosen]
    else:
        q[state] = dict()
        for action in [0, 1]: # TODO Fix (generic) action space
            q[state][action] = 0.0
        return random.randint(0, 1)

def state_from_space(space):
    # print(np.ndarray.round(space, 1))
    return str(np.ndarray.round(space, 1))

# sys.exit()

# Running X episodes
X = 1000
for i_episode in range(X):

    # Resetting environment
    observation = env.reset()
    old_state = state_from_space(observation)

    # Running through an episode
    for t in range(100):
        env.render()

        # Select action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = max_a(old_state)

        # Perform action, observe reward and new state
        observation, reward, done, info = env.step(action)
        if not old_state in q:
            q[old_state] = dict()
        new_state = state_from_space(observation)
        if not new_state in q:
            q[new_state] = dict()
            for action in [0, 1]: # TODO Same as above
                q[new_state][action] = 0.0

        # Q-learning equation
        q[old_state][action] = (1 - alpha) * q[old_state][action] + alpha * (reward + gamma * q[new_state][max_a(new_state)])
        # print(q[old_state][action])
        
        # Updating state
        old_state = new_state

        # Aborting episode if done
        if done:
           print("Episode finished after {} timesteps".format(t+1))
           break

sys.exit()
