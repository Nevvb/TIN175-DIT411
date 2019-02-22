import sys
import random
import gym
from gym import spaces


# --- Specific cartpole q-learning

env = gym.make('CartPole-v0')

q = dict()

size_up = 10
actions = env.action_space

# print("actions: "+str(actions))

e_start = 0.5
e_deacc = 0.0001
e_min = 0.01
epsilon = e_start   # Epsilon greedy probability
alfa = 0.5          # Learning rate
gamma = 1.0         # Discount factor

def max_a(state):
    reward = 0
    best_action = actions.sample()
    s0 = state[:-1]
    for i in range(actions.n):
        s = s0 + str(i)
        if get_reward(s) > reward:
            reward = get_reward(s)
            best_action = get_action(s)
    #         print("Best action, state: "+s+", action: "+str(get_action(s))+", reward: "+str(get_reward(s)))
    #     else:
    #         print("Best action, state: "+s+", action: "+str(get_action(s)))
    # print("Best action - result: "+str(best_action))
    return best_action

def get_reward(state):
    reward = 0
    if state in q:
        reward = q[state]
        # print("Got reward!!!: "+str(reward)+", state: "+str(state))
    else:
        q[state] = reward
    return reward

def get_action(state):
    action = int(state[-1:])
    # print("Got action! "+str(action)+", state: "+str(state))
    return action

def create_state(observation, action):
    s = ""
    for o in observation:
        n = size_up * o
        s = s + str(int(round(n))) + "_"
    s = s + str(action)
    # print("Create state: "+s+", action: "+str(action))
    return s


# sys.exit()

# Running X episodes
X = 10000
for i_episode in range(X):

    # Resetting environment
    observation = env.reset()
    old_state = create_state(observation, actions.sample())
    new_state = old_state

    # Running through an episode
    for t in range(200):
        env.render()

        # print("old state reward: "+str(old_reward)+ ", new reward: "+str(q[old_state]))

        # Select action
        r = random.uniform(0.0, 1.0)
        # print("Explore or exploit? "+str(r)+" (epsilon: "+str(epsilon)+")")
        if (r > epsilon):
            action = max_a(old_state)
        else:
            action = actions.sample()
        if epsilon > e_min:
            epsilon -= e_deacc

        # Perform action, observe reward and new state
        observation, reward, done, info = env.step(action)
        new_state = create_state(observation, action)

        # print("Old: "+str(old_state)+", reward: "+str(get_reward(old_state)))
        # print("New: "+str(new_state)+", reward: "+str(get_reward(new_state)))

        # Q-learning equation
        best_new_state = create_state(observation, max_a(new_state))
        best_new_reward = get_reward(best_new_state)
        old_reward = get_reward(old_state)
        # print("Best: "+str(best_new_state)+", reward: "+str(best_new_reward))
        new_reward = old_reward + alfa * (reward + gamma * best_new_reward - old_reward)
        q[old_state] = new_reward
        # print("old reward: "+str(old_reward)+ ", new reward: "+str(get_reward(old_state)))

        # Updating state
        old_state = new_state

        # Aborting episode if done
        if done:
           print("Episode "+str(i_episode)+" finished after {} timesteps".format(t+1))
           break


# Forever loop
while True:
    i_episode += 1

    # Resetting environment
    observation = env.reset()

    # Running through an episode
    for t in range(200):
        env.render()

        # Select action
        action = max_a(old_state)

        # Perform action
        observation, reward, done, info = env.step(action)

        # Aborting episode if done
        if done:
           print("(Forever loop) Episode "+str(i_episode)+" finished after {} timesteps".format(t+1))
           break


sys.exit()