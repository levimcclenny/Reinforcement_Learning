#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 02:10:33 2017

@author: levimcclenny
"""


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.violinplot(data = np.random.normal(0,1,10) + np.random.randn(200, 10), color = 'gray')

class Bandit:
    def __init__(self):
        self.q_star= np.random.normal(0,1,10)
        self.counts = np.zeros(10)
        self.q_est = np.zeros(10)
        
    def get_reward(self,action):
        reward = self.q_star[action] + np.random.normal(0,1)
        self.counts[action] += 1
        self.q_est[action] += (1/self.counts[action])*(reward - self.q_est[action])
        return reward
    
    def get_action(self,eps):
        if eps > np.random.random(): 
            return np.random.randint(10)
        else:
            return np.argmax(self.q_est)
    
def simulate(bandit, time, eps):
    rew = []
    for j in range(0,time):
        action = bandit.get_action(eps)
        reward = bandit.get_reward(action)
        rew.append(reward)
    return rew

final = [np.zeros(1000), np.zeros(1000), np.zeros(1000)]
epsilons = [0, .01, .1]
for k in range(len(epsilons)):  
    for i in range(2000):
        bandit = Bandit()
        final[k] += simulate(bandit, 1000, epsilons[k])

for i in range(len(final)):
    plt.plot(final[i]/2000, label = "eps = " +str(epsilons[i]))
plt.legend()




final_reward = [[], [], []]
optimal_action = [[], [], []]
eps = [0, .01, .1]
for k in range(len(eps)):
    Rewards = np.zeros(1000)
    OptimalAction = np.zeros(1000)
    for i in range(2000):
        q_star = np.random.normal(0,1,10)
        bestAction = np.argmax(q_star)
        q_est = np.zeros(10)
        counts = np.zeros(10)
        rew = []
        #optimal_count = 0
        optimalAction = []
        for j in range(0,1000):
            optimal_count = 0
            if eps[k] > np.random.random(): 
                action = np.random.randint(10)
            else:
                action = np.argmax(q_est)
            if action == bestAction:
                optimal_count = 1

            noise = np.random.normal(0,1)
            reward = q_star[action] + noise
            counts[action] +=1
            q_est[action] += (1/counts[action])*(reward - q_est[action])
            rew.append(reward)
            optimalAction.append(optimal_count)
        Rewards += rew
        OptimalAction += optimalAction
    
    final_reward[k] = Rewards/2000
    optimal_action[k] = OptimalAction/(2000)
    
plt.figure(1)
plt.subplot(211)
for i in range(len(final_reward)):
    plt.plot(final_reward[i], label = "eps = " +str(eps[i]))
plt.legend()
plt.xlabel('Steps')
plt.ylabel('average reward')
plt.legend()

plt.subplot(212)
for i in range(len(optimal_action)):
    plt.plot(optimal_action[i], label = "eps = " +str(eps[i]))
plt.xlabel('Steps')
plt.ylabel('% optimal action')
plt.legend()


