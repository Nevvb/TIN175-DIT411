import sys
import gym

from gym import spaces

# --- State class for q-learning

class State:
    def __init__(self, state_list, action):
        self.states = state_list
        self.size = len(self.states)
        self.action = action

    def __hash__(self):
        h = 0;
        for value in self.states:
            h = h*31 + value
        h = h*37 + self.action
        return int(round(h))

    def __str__(self):
        return "State: " + str(self.states) + ", action: " + str(self.action)

    def copy(self):
        new_state = State(self.states, self.action)
        return new_state

# --- Specific cartpole q-learning

env = gym.make('CartPole-v0')

q = dict()

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
gamma = 0.8     # Discount factor

def max_a(state):
    s = state.copy()
    reward = 0
    best_action = actions.sample()
    for i in range(actions.n):
        s.action = i
        if s in q and q[s] > reward:
            reward = q[s]
            best_action = i
    return best_action

def get_reward(state):
    reward = 0
    if state in q:
        reward = q[state]
    return reward

def state_from_space(space, action):
    l = list()
    for i in range(len(space)):
        n = space[i]
        # for x in range(high_multi[i]):
        #     n *= 10
        n *= size_up
        l.append(int(round(n)))
    state = State(l, action)
    return state

# sys.exit()

# Running X episodes
X = 100
for i_episode in range(X):

    # Resetting environment
    observation = env.reset()
    old_state = state_from_space(observation, actions.sample())
    new_state = old_state.copy()

    # Running through an episode
    for t in range(100):
        env.render()

        # print("old state reward: "+str(old_reward)+ ", new reward: "+str(q[old_state]))

        # Select action
        action = max_a(old_state)
        # if old_state in q:
        #     print("in q!")
        #     action = q[old_state].action
        # else:
        #     action = env.action_space.sample()

        # Perform action, observe reward and new state
        observation, reward, done, info = env.step(action)
        new_state = state_from_space(observation, action)

        print("Old: "+str(old_state)+", reward: "+str(get_reward(old_state)))
        print("New: "+str(new_state)+", reward: "+str(get_reward(new_state)))

        # Q-learning equation
        best_new_state = State(new_state.states, max_a(new_state))
        best_new_reward = get_reward(best_new_state)
        old_reward = get_reward(old_state)
        # print("Best: "+str(best_new_state)+", reward: "+str(best_new_reward))
        q[old_state] = old_reward + alfa * (reward + gamma * best_new_reward - old_reward)
        print("old reward: "+str(old_reward)+ ", new reward: "+str(q[old_state])+", vs function 'get_reward': "+str(get_reward(old_state)))

        # Updating state
        old_state = new_state.copy()

        # Aborting episode if done
        if done:
           print("Episode finished after {} timesteps".format(t+1))
           break

sys.exit()