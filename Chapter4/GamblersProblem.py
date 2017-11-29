#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:19:52 2017

@author: levimcclenny
"""

import numpy as np
import matplotlib.pyplot as plt


def getreward(state, action, value, Goal):
    if state + action == Goal:
        reward = 1
    else:
        reward = value[state + action]
    return reward


def gambler(goal, p):
    values = np.zeros(101)
    values[100] = 1
    delta = 1
    
    sweeps = []
    while delta > 1e-18:
        #instances.append(values)
        delta = 0.0
        newvalue = np.zeros(101)
        for state in np.arange(1,goal+1):
            rewards = []
            actions = np.arange(min((goal-state), state) + 1)
            for action in actions:
                rewards.append(p * getreward(state, action, values, goal) + 
                                           (1-p) * getreward(state, -action, values, goal))
            newvalue[state] = np.max(rewards)
        delta = np.sum(np.abs(newvalue - values))
        values = newvalue
        sweeps.append(values)
        
    return values, sweeps

Values, Sweeps = gambler(100, .4)

plt.plot(Sweeps[0], label = 'sweep 1')
plt.plot(Sweeps[1], label = 'sweep 2')
plt.plot(Sweeps[2], label = 'sweep 3')
plt.plot(Sweeps[3], label = 'sweep 4')
plt.plot(Values, label = 'Final')
plt.legend()
plt.show()


Values, Sweeps = gambler(100, .25)

plt.plot(Sweeps[0], label = 'sweep 1')
plt.plot(Sweeps[1], label = 'sweep 2')
plt.plot(Sweeps[2], label = 'sweep 3')
plt.plot(Sweeps[3], label = 'sweep 4')
plt.plot(Values, label = 'Final')
plt.legend()
plt.show()
    
Values, Sweeps = gambler(100, .55)

for i in (0,1,2,3,9,99):
    Label = 'Sweep '+str(i+1)
    plt.plot(Sweeps[i], label = Label)
plt.plot(Values, label = 'Final')
plt.legend()
plt.show()