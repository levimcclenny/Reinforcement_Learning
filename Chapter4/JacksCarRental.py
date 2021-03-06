#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:54:17 2017

@author: levimcclenny
"""

import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt

# current policy
policy = np.zeros((21,21))
value = np.zeros((21,21))
# all possible actions
actions = np.arange(-5, 6)

#generate lookup tables for poisson distributions
lamina3 = []
lamina4 = []
for a in range(0,21):
    lamina3.append(poisson.pmf(a,3))
    lamina4.append(poisson.pmf(a,4))


def getreward(inputstate, inputaction, inputvalue):
    reward = float(0)
    for requestA in range(0, 21):
        for requestB in range(0, 21):
            carsA = int(inputstate[0] - inputaction)
            if carsA > 20:
                carsA = 20
                
            carsB = int(inputstate[1] + inputaction)
            if carsB > 20:
                carsB = 20
            
            if requestA > carsA:
                actualA = carsA
            else:
                actualA = requestA
            
            if requestB > carsB:
                actualB = carsB
            else:
                actualB = requestB
            
            profit = (actualA + actualB) * 10
            
            carsA -= actualA
            carsB -= actualB
            
            #Return cars
            #Here we assume that the number of cars returning is the expected value of the
            #poisson(3) and poisson(2) distributions, which saves signifigantly on 
            #computational intensity.
            carsA += 3
            if carsA > 20:
                carsA = 20
            carsB += 2
            if carsB >20:
                carsB = 20
            
            reward += lamina3[requestA] * lamina4[requestB] * (profit + .9 * inputvalue[carsA, carsB])
            
    #ubtract cost of moving cars
    reward += -2 * abs(inputaction)
    return reward


policy = np.zeros((21,21))
value = np.zeros((21,21))
# all possible actions
actions = np.arange(-5, 6)



newvalue= np.zeros((21, 21))
improve = False
pi= 0
policies = []
policy_stable = False
while not policy_stable:
    # Policy Evaluation
    for a in range(21):
        for b in range(21):
           newvalue[a, b] = getreward([a, b], policy[a, b], value)
    if np.sum(np.abs(newvalue - value)) < .0001:
        value[:] = newvalue
        improve = True
    value[:] = newvalue
 
    # policy Improvement
    if improve == True:
        policies.append(policy)
        newpolicy = np.zeros((21, 21))
        for i in range(21):
            for j in range(21):
                returns = []
                # go through all actions and select the best one
                for action in actions:
                    returns.append(getreward([i, j], action, value))
                newpolicy[i, j] = actions[np.argmax(returns)]
        if np.sum(newpolicy != policy) == 0:
            policy_stable = True
        policy = newpolicy
        improve = False

fig, axn = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 10))

for i, ax in enumerate(axn.flat):
    sns.heatmap(policies[i+1], ax=ax, vmin=-4, vmax=6, 
                cmap="YlGnBu", annot = True, cbar = False)
    ax.invert_yaxis()
    ax.set_title("$\pi_{"+ str(i+1) + "}$" )
plt.show()

ax = sns.heatmap(value, vmin = 400, vmax = 620)
ax.invert_yaxis()
plt.show()
    
    

