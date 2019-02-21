import sys
import gym

from gym import spaces

# --- State class for q-learning

class State:
    def __init__(self, state_list):
        self.size = len(state_list)
        self.states = state_list

    def __hash__(self):
        h = 0;
        for value in states:
            h = hash*31 + hash(value)
        return h

    def __str__(self):
        return "State: " + str(self.states)

# --- Specific cartpole q-learning

env = gym.make('CartPole-v0')

q = dict()
# for s in range(env.observation_space.n)
#     for a in range(env.action_space.n)
a = {1, 2, 3, 4}
b = State(a)
# q[1] = "hej"
print(b)

sys.exit()

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # env.step(env.action_space.sample()) # take a random action
        print(observation);
        #action = env.action_space.sample()
        action = 0
        observation, reward, done, info = env.step(action)
        #if done:
        #    print("Episode finished after {} timesteps".format(t+1))
        #    break
sys.exit()