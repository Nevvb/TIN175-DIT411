import sys
import gym
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
#env = gym.make('MsPacman-v0')
#env = gym.make('LunarLander-v2')
#env = gym.make('SpaceInvaders-v0')
#env = gym.make('FrozenLake-v0')
env.reset()
for _ in range(20):
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    env.reset()
sys.exit()