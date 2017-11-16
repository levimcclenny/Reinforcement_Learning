#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 02:10:33 2017

@author: levimcclenny
"""


import numpy as np
import matplotlib.pyplot as plt

def gradient_bandit(arms, time, runs, step_size, avgreward = 4, 
                    GradientBase = True, Stationary = True):
    final_reward = [] 
    optimal_action = [] 
    OptimalAction = np.zeros(time)
    Rewards = np.zeros(time)
    for i in range(runs):
        q_star = np.random.normal(avgreward,1,arms)
        bestAction = np.argmax(q_star)
        H = np.zeros(arms)
        counts = np.zeros(arms)
        rew = []
        pi = np.zeros(arms)
        #optimal_count = 0
        optimalAction = []
        Rbar = 0.0
        for j in range(0,time):
            optimal_count = 0
            pi = np.exp(H)/np.sum(np.exp(H))
    
            action = np.random.choice(arms, p=pi)
            if action == bestAction:
                optimal_count = 1
            reward = q_star[action] + np.random.normal(0,1)
            counts[action] +=1
            Rbar = (j)/(j+1)*Rbar + reward/(j+1)
            ones = np.array(np.repeat(1,arms))
            ones[action] = 1
            one = np.zeros(arms)
            one[action] = 1
            if GradientBase:
                baseline = Rbar
            else:
                baseline = 0
            H += step_size*(reward - baseline)*(one-pi)
            rew.append(reward)
            optimalAction.append(optimal_count)
            if not Stationary:
                q_star += np.random.normal(0,0.01,arms)
                bestAction = np.argmax(q_star)
        Rewards += rew
        OptimalAction += optimalAction
    
    final_reward = Rewards/runs
    optimal_action = OptimalAction/runs
    return final_reward, optimal_action

final_reward1, optimal_action1 = gradient_bandit(10, 1000,1000, .1)
final_reward2, optimal_action2 = gradient_bandit(10, 1000,1000, .4)
final_reward3, optimal_action3 = gradient_bandit(10, 1000,1000, .1, GradientBase = False)
final_reward4, optimal_action4 = gradient_bandit(10, 1000,1000, .4, GradientBase = False)


plt.plot(optimal_action1, label = r"$\alpha = 0.1$, baseline")
plt.plot(optimal_action2, label = r"$\alpha = 0.4$, baseline")
plt.plot(optimal_action3, label = r"$\alpha = 0.1$, w/o baseline")
plt.plot(optimal_action4, label = r"$\alpha = 0.4$, w/o baseline")
plt.xlabel('Steps')
plt.ylabel('% optimal action')
plt.legend()


