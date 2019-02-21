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
action_space = [0, 1]
print("actions: "+str(actions))

epsilon = 0.1   # Epsilon greedy probability
alpha = 1.0     # Learning rate
gamma = 0.94    # Discount factor

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
        action_chosen = random.randint(0, len(best_actions) - 1)
        # print(f'action chosen: {best_actions[action_chosen]}')
        return best_actions[action_chosen]
    else:
        q[state] = dict()
        for action in action_space: # TODO Fix (generic) action space
            q[state][action] = 0.0
        return random.randint(0, len(action_space) - 1)

def state_from_space(space):
    # print(np.ndarray.round(space, 1))
    return str(np.ndarray.round(10*space, 0)[1:3])

# sys.exit()

# Running X episodes
X = 1000
for i_episode in range(X):

    # Resetting environment
    observation = env.reset()
    old_state = state_from_space(observation)
    epsilon *= 0.8
    print(f'epsilon={epsilon}')

    # Running through an episode
    for t in range(1000):
        env.render()

        # Select action
        action = max_a(old_state)
        if random.random() < epsilon:
            action = env.action_space.sample()

        # Perform action, observe reward and new state
        observation, reward, done, info = env.step(action)
        reward = 1.0 - abs(4.0 * observation[2])
        print(f'reward: {reward}')
        # if not old_state in q:
            # q[old_state] = dict()
            # for action in action_space: # TODO Same as above
                # q[new_state][action] = 0.0
        new_state = state_from_space(observation)
        if not new_state in q:
            q[new_state] = dict()
            for action in action_space: # TODO Same as above
                q[new_state][action] = 0.0

        # Q-learning equation
        print(f'state size: {len(q)}')
        q[old_state][action] = (1 - alpha) * q[old_state][action] + alpha * (reward + gamma * q[new_state][max_a(new_state)])
        # print(q[old_state][action])
        
        # Updating state
        old_state = new_state

        # Aborting episode if done
        if done:
           print("Episode finished after {} timesteps".format(t+1))
           break

sys.exit()
