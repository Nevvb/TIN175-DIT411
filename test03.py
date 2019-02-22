import sys
import gym

from gym import spaces

# # --- State class for q-learning

# class State:
#     def __init__(self, state_list, action):
#         self.states = state_list
#         self.size = len(self.states)
#         self.action = action

#     def __hash__(self):
#         h = 0;
#         for value in self.states:
#             h = h*31 + value*10
#         h = h*37 + self.action
#         return int(round(h))

#     def __str__(self):
#         return "State: " + str(self.states) + ", action: " + str(self.action)

#     def copy(self):
#         new_state = State(self.states, self.action)
#         return new_state

# # --- Q object
# class Q:
#     def __init__(self):
#         self.q = dict()

#     def put(self, state, reward):
#         self.q[state] = reward

#     def get_reward(self, state):
#         reward = 0
#         if state in self.q:
#             reward = self.q[state]
#             print("Got reward!!!: "+str(reward)+", state: "+str(state))
#         else:
#             self.q[state] = 0
#         return reward

#     def contains(self, state):
#         if state in self.q:
#             return True
#         return False

# --- Specific cartpole q-learning

env = gym.make('CartPole-v0')

q = dict()
# q_action = dict()

# q = Q()

size_up = 10
# high = env.observation_space.high
# high_multi = list()
# low = env.observation_space.low
# low_multi = list()
actions = env.action_space

# for i in range(len(high)):
#     high_multi.append(0)
#     while abs(high[i]) < 10:
#         high[i] *= 10
#         high_multi[i] += 1
#     high[i] = round(high[i])

# for i in range(len(low)):
#     low_multi.append(0)
#     while abs(low[i]) < 10:
#         low[i] *= 10
#         low_multi[i] += 1
#     low[i] = round(low[i])

# print("max: "+str(high))
# print("min: "+str(low))
print("actions: "+str(actions))

epsilon = 0.9   # Epsilon greedy probability
alfa = 0.5      # Learning rate
gamma = 1.0     # Discount factor

def max_a(state):
    # s = state.copy()
    reward = 0
    best_action = actions.sample()
    s = state[:-1]
    for i in range(actions.n):
        # q_action(state) = i
        s = s + str(i)
        if s in q and get_reward(s) > reward:
            reward = q[s]
            best_action = get_action(s)
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
    # action = actions.sample()
    # if state in q_action:
    #     action = q_action[state]
    #     # print("Got reward!!!: "+str(reward)+", state: "+str(state))
    # else:
    #     q_action[state] = action
    # return action
    return int(state[-1:])

def create_state(observation, action):
    s = ""
    for o in observation:
        n = size_up * o
        s = s + "-" + str(int(round(n)))
    s = s + "-" + str(action)
    return s

# def save_state(state, action, reward):
#     q[state] = reward
    # q_action[state] = action

# def state_from_space(space, action):
#     l = list()
#     for i in range(len(space)):
#         n = space[i]
#         # for x in range(high_multi[i]):
#         #     n *= 10
#         n *= size_up
#         l.append(int(round(n)))
#     state = State(l, action)
#     return state

# sys.exit()

# Running X episodes
X = 100
for i_episode in range(X):

    # Resetting environment
    observation = env.reset()
    # old_state = state_from_space(observation, actions.sample())
    old_state = create_state(observation, actions.sample())
    new_state = old_state
    # print("hash: "+str(old_state.__hash__())+ ", hash2: "+str(old_state.__hash__()))

    # Running through an episode
    for t in range(100):
        env.render()

        # print("old state reward: "+str(old_reward)+ ", new reward: "+str(q[old_state]))

        # Select action
        action = max_a(old_state)

        # Perform action, observe reward and new state
        observation, reward, done, info = env.step(action)
        # new_state = state_from_space(observation, action)
        new_state = create_state(observation, action)

        # print("Old: "+str(old_state)+", reward: "+str(get_reward(old_state)))
        # print("New: "+str(new_state)+", reward: "+str(get_reward(new_state)))

        # Q-learning equation
        # best_new_state = State(new_state.states, max_a(new_state))
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
           print("Episode finished after {} timesteps".format(t+1))
           break

sys.exit()