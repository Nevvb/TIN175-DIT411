import sys
import gym
import keyboard

from gym import spaces

# --- State class for q-learning

class State:
    def __init__(self, state_list):
        self.states = state_list
        self.size = len(self.states)

    def __hash__(self):
        h = 0;
        for value in self.states:
            h = h*31 + value
        return int(round(h))

    def __str__(self):
        return "State: " + str(self.states)

# --- Specific cartpole q-learning

env = gym.make('CartPole-v0')

q = dict()

high = env.observation_space.high
low = env.observation_space.low

for i in range(len(high)):
    while abs(high[i]) < 10:
        high[i] *= 10
    high[i] = round(high[i])

for i in range(len(low)):
    while abs(low[i]) < 10:
        low[i] *= 10
    low[i] = round(low[i])

print("max: "+str(high))
print("min: "+str(low))

epsilon = 0.9

# sys.exit()

for i_episode in range(20):
    observation = env.reset()
    s1 = State(observation)
    s2 = s1
    for t in range(100):
        # if keyboard.is_pressed('esc'):
        #     print("quit!")
        #     break
        env.render()

        print(s1)
        if s1 in q:
            print("in q!")
            action = q[s1]
        else:
            action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        if done:
           print("Episode finished after {} timesteps".format(t+1))
           break

        s2 = s1
        s1 = State(observation)

sys.exit()