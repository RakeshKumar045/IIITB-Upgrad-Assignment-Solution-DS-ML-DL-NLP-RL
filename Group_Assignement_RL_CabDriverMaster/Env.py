# Import routines

import random

import numpy as np

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [[i, j] for i in range(m) for j in range(m)]
        self.state_space = [[i, j, k] for i in range(m) for j in range(t) for k in range(d)]
        self.state_init = [np.random.randint(0, m - 1), np.random.randint(0, t - 1), np.random.randint(0, d - 1)]

        # Start the first round
        self.reset()
        self.maxDay = 30
        self.maxTime = 24 * 30
        self.actionSize = (m - 1) * m + 1
        self.totalTime = 0;

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        # state_encod = np.array(state).reshape(state,-1)
        return np.reshape(state, -1)

    # Use this function if you are using architecture-2
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

    #     return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        requests = 0
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m - 1) * m + 1),
                                               requests)  # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append([0, 0])

        return possible_actions_index, actions

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        start_loc, time, day = state
        pickup, drop = action
        # Reward cal formula
        # 𝑅(𝑠 = 𝑋𝑖𝑇 𝑗𝐷𝑘) = {
        # 𝑅𝑘 ∗ (𝑇𝑖𝑚𝑒(𝑝,𝑞)) − 𝐶𝑓 ∗ (𝑇𝑖𝑚𝑒(𝑝,𝑞) + 𝑇𝑖𝑚𝑒(𝑖,𝑝)) 𝑎 = (𝑝,𝑞), −𝐶𝑓 𝑎 = (0,0) } 
        if action == [0, 0]:
            return -C
        else:
            time_elapsed_till_pickup = Time_matrix[start_loc, pickup, int(time), int(day)]
            time_next = np.int((time + time_elapsed_till_pickup) % t)
            day_next = np.int((day + (time + time_elapsed_till_pickup) // t) % d)
            timepicktodrop = Time_matrix[pickup, drop, int(time_next), int(day_next)]
            R_Cost = R * timepicktodrop
            C_Cost = C * (timepicktodrop + Time_matrix[start_loc, pickup, int(time), int(day)])
            reward = R_Cost - C_Cost
            return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        currentloc, time, day = state
        pickup, drop = action
        # when action is (0,0)                           
        if action == [0, 0]:
            time_elapsed = 1
            self.totalTime = self.totalTime + time_elapsed
            time_next = np.int((time + time_elapsed) % t)
            day_next = np.int((day + (time + time_elapsed) // t) % d)
        else:
            # when pickup loc same as current location, current time and day could change                       
            if currentloc == pickup:
                time_elapsed = Time_matrix[pickup, drop, int(time), int(day)]
                self.totalTime = self.totalTime + time_elapsed
                time_next = np.int((time + time_elapsed) % t)
                day_next = np.int((day + (time + time_elapsed) // t) % d)
            else:
                # when pickup is not same as current location, current time and day could change 
                time_elapsed_till_pickup = Time_matrix[currentloc, pickup, int(time), int(day)]
                time_next_temp = (time + time_elapsed_till_pickup) % t
                day_next_temp = (day + (time + time_elapsed_till_pickup) // t) % d
                time = time_next_temp
                day = day_next_temp
                time_elapsed = Time_matrix[pickup, drop, int(time), int(day)]
                self.totalTime = self.totalTime + time_elapsed + time_elapsed_till_pickup
                time_next = np.int((time + time_elapsed) % t)
                day_next = np.int((day + (time + time_elapsed) // t) % d)
        # check whether it is a terminal state 
        if (self.totalTime >= self.maxTime):
            terminal_state = True
            self.totalTime = 0
        else:
            terminal_state = False
        # returns terminal state as True or False 
        next_state = (drop, time_next, day_next)
        return next_state, terminal_state

    def reset(self):
        return self.action_space, self.state_space, self.state_init
