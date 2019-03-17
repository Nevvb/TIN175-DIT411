=== README.txt

Our project has focused on two tasks mainly: optimizing Q-learning for CartPole, and trying to make it work for Atari's Breakout. We therefore submit two files dealing with these two problems separately.


=== main.py
Our main implementation of Q-learning for OpenAI-Gym's CartPole-v1 program.
Most work on optimization has been in this program, should work fairly good.

Packages needed:
    sys
    gym
    random
    math
    numpy
    matplotlib
Run:
    Simply run file through python, should be selfcontained


=== atari.py
Implements simple Q-learning for Atari's Breakout through OpenAI-Gym.
Is currently not working very well, due to much too big state space.

Packages needed:
    gym
    random
    matplotlib
    numpy
    time
    datetime
Run:
    Simply run file through python, should be selfcontained