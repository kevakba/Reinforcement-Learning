# Import routines
import numpy as np
import math
import random
from itertools import product

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""

        prod = product([1,2,3,4,5],[1,2,3,4,5])
        possible_actions = []
        for i in prod:
            if i[0]!=i[1]:
                possible_actions.append(i)
        self.action_space = possible_actions

        self.state_init = [np.random.randint(1,6), np.random.randint(1,25), np.random.randint(1,8)]
        
        self.week = 0  #to keep track of terminal state
        
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        M = np.zeros((1,m))
        M[0, int(state[0]-1)] = 1

        T = np.zeros((1,t))
        T[0, int(state[1]-1)] = 1

        D = np.zeros((1,d))
        D[0, int(state[2]-1)] = 1
        
        state_encod = np.concatenate((M,T,D), axis=None)
        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 1:
            requests = np.random.poisson(2)
        elif location == 2:
                requests = np.random.poisson(12)
        elif location == 3:
                requests = np.random.poisson(4)
        elif location == 4:
                requests = np.random.poisson(7)
        elif location == 5:
                requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(0, (m-1)*m), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        
        actions.append((0,0))   #appending action (0,0)
        possible_actions_index.append(20)  #appending index of action (0,0)

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        time_p_q = Time_matrix[int(action[0]-1)][int(action[1]-1)][int(state[1]-1)][int(state[2]-1)]
        time_i_p = Time_matrix[int(state[0]-1)][int(action[0]-1)][int(state[1]-1)][int(state[2]-1)]

        if (action[0]==0) and (action[1]==0):
            reward = -C
        else:
            reward = R*time_p_q - C*(time_p_q + time_i_p)
        return reward



    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        if (action[0]!=0) and (action[1]!=0):
            time_i_p = Time_matrix[int(state[0]-1)][int(action[0]-1)][int(state[1]-1)][int(state[2]-1)]
            time_p_q = Time_matrix[int(action[0]-1)][int(action[1]-1)][int(state[1]-1)][int(state[2]-1)]
            time = state[1] + time_i_p + time_p_q
            day = state[2]
            if time > 24:
                time = time - 24
                day = day + 1

            if day>7:
                day = 1
                self.week+=1
            next_state = [action[1], time, day]    #going to the next state
        else:
            time = state[1] + 1
            day = state[2]
            if time > 24:
                time = time - 24
                day = day + 1

            if day>7:
                day = 1
                self.week+=1
            next_state = [state[0], time, day]    # increasing time by 1 hr for action (0,0)
   
        return next_state


            
    def reset(self):
        return self.state_init  # self.action_space, self.state_space
