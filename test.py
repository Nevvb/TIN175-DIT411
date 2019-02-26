import gym
import numpy as np


cartP = [-.3, -.2 , -.1, 0, .1, .2, .3]
cartV = [-3, -2, -1, 0, 1, 2, 3]
poleV = [-3, -2, -1, 0, 1, 2, 3]
angle = [-.2, -.1, -.05, 0, .05, .1, .2]


def disc(s):
	#print(s[2])
	a = 0
	p = 0

	i = 0
	for i in range(len(angle)):
		if angle[i] > s[2]:
			a = i
			break

	i = 0
	for i in range(len(poleV)):
		if poleV[i] > s[3]:
			p = i
			break

	#print(p , a)
	return [p,a]
env = gym.make('CartPole-v0')
env.reset()
env.render()

print('Action space: {}'.format(env.action_space))
print(env.observation_space.shape[0])

Q = np.zeros((7,7,2))

print(Q[4,1])



alpha = 0.6
gamma = 0.1
epis = 300
rev_list = []


for i in range(epis):
	#print("Episode %i", i)
	s = env.reset()
	#print(s)
	rAll = 0
	done = False
	j = 0
	kk = 1000
	while j < kk:
		
		if i % 50 == 0:
			env.render()
			#print(s[3])
		j+=1

		ds = disc(s)

		a = np.argmax(Q[ds[0],ds[1], :] + np.random.randn(1,env.action_space.n)*(1./((i)+1)))
		#print(a)
		s1,r,done,t = env.step(a)
		#print(s1)
		ds1 = disc(s1)
		rew = max(0, 1 - (abs(s[2])/2 + abs(s[3]) * 2.5))
		#gamma = i/epis
		Q[ds[0],ds[1], a] = Q[ds[0],ds[1], a] + alpha*( rew + gamma*np.max(Q[ds1[0], ds1[1],a]) - Q[ds[0],ds[1], a])
		rAll += r
		if done:
			break
		s = s1

	rev_list.append(rAll)
	#env.render()



print "Reward Sum on all episodes " + str(sum(rev_list[-100:])/100)
print "Final Values Q-Table"
#print Q
