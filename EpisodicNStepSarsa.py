
# Sarsa is a value iteration algorithm that learns Q(s,a)
# Combined with some policy that is based on Q-values (like eps-greedy)

# Task that is somewhat problem dependent in linear SARSA is how to pick the right features
# Specifically, these features should predict the value of a particular action given the state description by features
# These features could just be the state description, but also should probably be augmented.
# Polynomial features: this is adding polynomial terms and interaction terms

# References and Plan
# Can use the textbook RLbook2020.pdf to get pseudocode
# Algorithm: n-step semi-gradient SARSA (pg 244)
# This algorithm is not going to be truly online. Will have to go

# Input: function approximation and featurization of observation space.
# Tilings of the position-velocity space give the feature vector x.
# q function is linear in these tiling features
# q function is exactly the form in Equation 10.3 on page 246

# Basically all of the work is in defining how q works and how the features work
# Other than this, can follow the algorithm exactly

# Need to understand exaclty the end of the algorithm and would like to understand, but prob not implement,
# The difference that eligibility traces would make- why is it not truly online without eligibility traces?

# Write out the whole algorithm here, with classes for the Q function approximation and the policy

import click
import gym
import numpy as np
from math import log, ceil, exp
from tiles3 import IHT

from QFunctions import LinearTilingQApproximator

# python EpisodicNStepSarsa.py --episodes 1 --alpha 0.03 --init_epsilon 0.1 --eps_decay_factor 0.0 --n 1 --gamma 0.9 --tile_resolution 8 --n_tilings 8 --d 3 --render

@click.command()
@click.option('--episodes', default=10)
@click.option('--alpha', default=0.1)
@click.option('--init_epsilon', default=0.1)
@click.option('--eps_decay_factor', default=0.1) # epsilon_t = init_epsilon*exp(-eps_decay_factor*t) so decay = 0 is constant
@click.option('--n', default=1)
@click.option('--gamma', default=0.5) # reward discounting factor
@click.option('--tile_resolution', default=8) # number of tiles in each dimension of feature space
@click.option('--n_tilings', default=8) # number of overlapping tilings
@click.option('--d', default=3) # either 2 if (position,velocity) and 3 if (position,velocity,acceleration)
@click.option('--render/--no-render', default=True)
def main(episodes, alpha, init_epsilon, eps_decay_factor,
         n, gamma, tile_resolution, n_tilings, d, render):
    # Instantiate the environment
    env = gym.make('MountainCar-v0')
    n_actions = 3 # Action space is 0,1,2 for mountain car

    # Initialize the hash table to store the tiling
    n_tiles = tile_resolution ** d * n_tilings * n_actions
    iht = IHT(2**(1+ceil(log(n_tiles, 2)))) # should be double the hash table size that we need

    # Initialize the Q function
    q_hat = LinearTilingQApproximator(iht, n_tilings, tile_resolution)

    # Initialize arrays to store actions, states and rewards
    # n-step SARSA means storing current info AND info for n more steps
    A = np.zeros(n+1, dtype=int)
    S = np.zeros((n+1,d)) # each row is either [position, velocity] or [position, velocity, acceleration]
    R = np.zeros(n+1)

    # Loop over episodes
    for episode in range(episodes):
        # Initial observation
        # For MountainCar, always starts with 0 velocity and append 0 acceleration
        observation = list(env.reset())

        if d == 2:
            S[0] = observation
        else:
            S[0] = observation + [0]

        # epsilon-greedy action based on initial state
        if np.random.uniform() <= init_epsilon:
            A[0] = env.action_space.sample()
        else:
            A[0] = np.argmax([q_hat(S[0], a) for a in range(n_actions)])

        # Set termination time to infinity to start
        # Initialize time counter
        t = 0
        T = np.inf

        # render
        if render: env.render()

        ### Print initial state
        print('t = {:d}'.format(t))
        print('S[t] = ' + np.array2string(S[0], precision=2))
        print('A[0] = {:d}'.format(A[0]))

        # Loop over time periods within an episode
        while True:

            # If we haven't terminated, then take an action
            # Store the next state and reward
            if t < T:
                observation, reward, done, info = env.step(A[t % (n+1)])
                if render: env.render()

                R[(t+1) % (n+1)] = reward
                if d == 2:
                    S[(t+1) % (n+1)] = list(observation)
                else:
                    S[(t+1) % (n+1)] = list(observation) + [observation[1] - S[t % (n+1), 1]]
                if done:
                    T = t + 1
                else:
                    epsilon = init_epsilon*exp(-eps_decay_factor*t)
                    if np.random.uniform() <= epsilon:
                        A[(t+1) % (n+1)] = env.action_space.sample()
                    else:
                        A[(t+1) % (n+1)] = np.argmax([q_hat(S[(t+1) % (n+1)], a) for a in range(n_actions)])

                ### Print state for t + 1
                print('After taking A[{:d}], the info for t+1 = {:d} is:'.format(t, t+1))
                print('t + 1 = {:d}'.format(t+1))
                print('S[t+1] = ' + np.array2string(S[(t+1) % (n+1)], precision=2))
                print('R[t+1] = {:f}'.format(R[(t+1) % (n+1)]))
                print('A[t+1] chosen based on S[t+1] and R[t+1].')
                print('A[t+1] = {:d}'.format(A[(t+1) % (n+1)]))

            # Set the period for which we are updating the weights
            # E.g. if n = 1, then can start updating at t = 0 because we have stored S_1 and R_1
            tau = t - n + 1

            # If we are ready to update the first state, then go ahead
            if tau >= 0:
                # discounted n-step return
                G = sum([gamma**(i-tau-1)*R[(i % (n+1))] for i in range(tau+1, min(tau+n, T) + 1)])
                # if you haven't terminated within n steps, then add the estimated return to go
                if tau + n < T:
                    G = G + gamma**n * q_hat(S[(tau+n) % (n+1)], A[(tau+n) % (n+1)])
                # Adjust the weights based on gradient of the error
                # The update function takes the state and the action to find the active tiles
                # Then updates each tile by alpha * error
                q_hat.update(S[tau % (n+1)],
                             A[tau % (n+1)],
                             alpha,
                             G - q_hat(S[tau % (n+1)], A[tau % (n+1)]))

            if tau == T - 1:
                print('Exiting at tau = {:d}'.format(tau))
                break

            t += 1


if __name__ == '__main__':
    main()
