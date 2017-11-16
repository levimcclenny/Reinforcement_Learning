#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 02:10:33 2017

@author: levimcclenny
"""


import numpy as np
import matplotlib.pyplot as plt



time = 10000
runs = 2000

def ten_bandit_testbed(time, runs, eps, stationary = True, 
                      constant_step = False, step_size = .01, perturb = False):
    final_reward = [[] for i in range(0,len(eps))]
    optimal_action = [[] for i in range(0,len(eps))]
    for k in range(len(eps)):
        Rewards = np.zeros(time)
        OptimalAction = np.zeros(time)
        for i in range(runs):
            q_star = np.random.normal(0,1,10)
            bestAction = np.argmax(q_star)
            q_est = np.zeros(10)
            counts = np.zeros(10)
            rew = []
            #optimal_count = 0
            optimalAction = []
            for j in range(0,time):
                optimal_count = 0
                if eps[k] > np.random.random(): 
                    action = np.random.randint(10)
                else:
                    action = np.argmax(q_est)
                if action == bestAction:
                    optimal_count = 1
                reward = q_star[action] + np.random.normal(0,1)
                counts[action] +=1
                if constant_step:
                    q_est[action] += step_size*(reward - q_est[action])
                if perturb:
                    q_est[action] += (1/counts[action])*(reward - q_est[action]) 
                    q_est += np.random.normal(0,.01,10)
                    
                else:
                    q_est[action] += (1/counts[action])*(reward - q_est[action])
                if not stationary:
                    q_star += np.random.normal(0,.01,10)
                    bestAction = np.argmax(q_star)
                rew.append(reward)
                optimalAction.append(optimal_count)
            Rewards += rew
            OptimalAction += optimalAction
        
        final_reward[k] = Rewards/runs
        optimal_action[k] = OptimalAction/runs
    return final_reward, optimal_action


reward1, action1 = ten_bandit_testbed(1000,2000, [0.1])
reward2, action2 = ten_bandit_testbed(1000,2000, [0.1], stationary = False)
reward3, action3 = ten_bandit_testbed(1000,2000, [0.1], stationary = False, constant_step = True)
reward4, action4 = ten_bandit_testbed(1000,2000, [0.1], stationary = False, perturb = True)

   
plt.figure(1)
plt.subplot(211)
plt.plot(reward1[0], label = "$\epsilon = .1$")
plt.plot(reward2[0], label = "Non-stationary")
plt.plot(reward3[0], label = "Constant step size")
plt.plot(reward4[0], label = "perturb")
plt.legend()
plt.xlabel('Steps')
plt.ylabel('average reward')
plt.legend()

plt.subplot(212)
plt.plot(action1[0], label = "$\epsilon = .1$")
plt.plot(action2[0], label = "Non-stationary")
plt.plot(action3[0], label = "Constant step size")
plt.plot(action4[0], label = "perturb")
plt.xlabel('Steps')
plt.ylabel('% optimal action')
plt.legend()


