import sys
import gym

from gym import spaces

# --- Different environments

# env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
# env = gym.make('MsPacman-v0')
# env = gym.make('LunarLander-v2')
# env = gym.make('SpaceInvaders-v0')
# env = gym.make('FrozenLake-v0')

# --- Test code

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

# space = spaces.Discrete(8)
# x = space.sample()
# assert space.contains(x)
# assert space.n == 8

# sys.exit()

# --- Specific cartpole q-learning

env = gym.make('CartPole-v0')

q = dict()
# for s in range(env.observation_space.n)
#     for a in range(env.action_space.n)
a = {1, 2, 3, 4}
q[1] = "hej"
print(q)

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