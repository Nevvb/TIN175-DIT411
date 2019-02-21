import gym
import numpy as np

env = gym.make("CartPole-v1")

Q = np.zeros([30, env.action_space.n])
alfa = 0.7 # learn rate
gamma = 0.4 # discount fact
epsilon = 0.99 # explore rate
n_episodes = 200

def get_action(state):
  state = discretize(state[2], state[1])
  return env.action_space.sample() if np.random.random() <= epsilon else np.argmax(Q[state])

def update_q(state, next_state, action, reward):
  state = discretize(state[2], state[1])
  next_state = discretize(next_state[2], next_state[1])
  Q[state][action] += alfa * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

def discretize(angle, velocity):
  i = 0
  if(abs(angle) > np.radians(12)):
    i -= 1
  if(angle > np.radians(-6)):
    i += 1
  if(angle > np.radians(-1)):
    i += 1
  if(angle > np.radians(0)):
    i += 1
  if(angle > np.radians(1)):
    i += 1
  if(angle > np.radians(6)):
    i += 1
  if(velocity < -0.5):
    i += 0
  elif(velocity < -0.1):
    i += 6
  elif(velocity < 0.1):
    i += 12
  elif(velocity < 0.5):
    i += 18
  else:
    i += 24
  return i

for e in range(n_episodes):
  state = env.reset()
  total_reward = 0
  done = False
  
  while not done:
    action = get_action(state)
    next_state, reward, done, _ = env.step(action)
    update_q(state, next_state, action, reward)
    total_reward += reward
    state = next_state
    epsilon = max(0.01, epsilon - 0.0001*e)
    #env.render()

  print("Episode: {}, total_reward: {:.2f}".format(e, total_reward)) 



    
