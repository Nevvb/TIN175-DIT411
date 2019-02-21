import sys
import gym
import random

import numpy as np
from gym import spaces

# --- State class for q-learning

class State:
    def __init__(self, state_list):
        self.states = state_list
        self.size = len(self.states)

    def __hash__(self):
        h = 0;
        for value in self.states:
            h = h*31 + value*10
        return int(round(h))

    def __str__(self):
        return "State: " + str(self.states)

    def copy(self):
        new_state = State(self.states)
        return new_state

# --- Specific cartpole q-learning

env = gym.make('CartPole-v0')

q = dict()

size_up = 10
actions = env.action_space
print("actions: "+str(actions))

epsilon = 0.01  # Epsilon greedy probability
alpha = 0.2     # Learning rate
gamma = 0.8     # Discount factor

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
        print(f'actions: {best_actions}, reward: {best_reward}')
        action_chosen = random.randint(0, len(best_actions) - 1)
        print(f'action chosen: {best_actions[action_chosen]}')
        return best_actions[action_chosen]
    else:
        q[state] = dict()
        for action in [0, 1]: # TODO Fix (generic) action space
            q[state][action] = 0.0
        return random.randint(0, 1)

def state_from_space(space):
    print(np.ndarray.round(space, 1))
    return str(np.ndarray.round(space, 1))
    # l = list()
    # for i in range(len(space)):
    #     n = space[i]
    #     # for x in range(high_multi[i]):
    #     #     n *= 10
    #     n *= size_up
    #     l.append(int(round(n)))
    # state = State(l)
    # return state

# sys.exit()

# Running X episodes
X = 100
for i_episode in range(X):

    # Resetting environment
    observation = env.reset()
    old_state = state_from_space(observation)
    # new_state = old_state.copy()
    # print("hash: "+str(old_state.__hash__())+ ", hash2: "+str(old_state.__hash__()))

    # Running through an episode
    for t in range(100):
        env.render()

        # print("old state reward: "+str(old_reward)+ ", new reward: "+str(q[old_state]))

        # Select action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = max_a(old_state)
        # if old_state in q:
        #     print("in q!")
        #     action = q[old_state].action
        # else:
        #     action = env.action_space.sample()

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
        print(q[old_state][action])
        

        # Updating state
        old_state = new_state
        # print(f'{q.q}\n')

        # Aborting episode if done
        if done:
           print("Episode finished after {} timesteps".format(t+1))
           # break

sys.exit()
