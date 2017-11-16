#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:06:13 2017

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

def k_bandit_testbed(arms, time, runs, eps, stationary = True, 
                      constant_step = False, step_size = .1, 
                      optimistic = False, opt_mean = 5,
                      UCB = False, c = 2):
    
    final_reward = [[] for i in range(0,len(eps))]
    optimal_action = [[] for i in range(0,len(eps))]
    for k in range(len(eps)):
        Rewards = np.zeros(time)
        OptimalAction = np.zeros(time)
        for i in range(runs):
            q_star = np.random.normal(0,1,arms)
            bestAction = np.argmax(q_star)
            q_est = np.zeros(arms)
            if optimistic:
                q_est = np.repeat(float(opt_mean), arms)
            counts = np.zeros(arms)
            rew = []
            #optimal_count = 0
            optimalAction = []
            for j in range(0,time):
                optimal_count = 0
                if eps[k] > np.random.random(): 
                    action = np.random.randint(arms)
                else:
                    action = np.argmax(q_est)
                if UCB:
                    action = np.argmax(q_est + c*np.sqrt(np.log(time)/(counts)))
                if action == bestAction:
                    optimal_count = 1
                reward = q_star[action] + np.random.normal(0,1)
                counts[action] +=1
                if constant_step:
                    q_est[action] += step_size*(reward - q_est[action])
                else:
                    q_est[action] += (1/counts[action])*(reward - q_est[action])
                if not stationary:
                    q_star += np.random.normal(0,.01,arms)
                    bestAction = np.argmax(q_star)
                rew.append(reward)
                optimalAction.append(optimal_count)
            Rewards += rew
            OptimalAction += optimalAction
        
        final_reward[k] = Rewards/runs
        optimal_action[k] = OptimalAction/runs
    return final_reward, optimal_action


rewards = []
for d in np.arange(-7,-1):
    print(d)
    rew, act = k_bandit_testbed(10, 1000, 2000, [(2**float(d))])
    rewards.append(sum(rew[0]))
eps = np.array(rewards)/1000
    

rewards = []
exp = np.arange(-5.,3)
C= 2**exp
for d in C:
    print(d)
    rew, act = gradient_bandit(10, 1000, 2000, d, avgreward = 0)
    rewards.append(sum(rew))
grad = np.array(rewards)/1000
plt.plot(grad)


rewards = []
exp = np.arange(-4.,3)
C= 2**exp
for d in C:
    print(d)
    rew, act = k_bandit_testbed(10, 1000, 2000, [0], constant_step = True, step_size = 0.10, UCB = True, c = d)
    rewards.append(sum(rew[0]))
UCB = np.array(rewards)/1000


rewards = []
exp = np.arange(-2.,3.)
C= 2**exp
for d in C:
    #print(d)
    rew, act = k_bandit_testbed(10, 1000, 2000, [0], constant_step = True, step_size = 0.10, optimistic = True, opt_mean = d)
    rewards.append(sum(rew[0]))
opt = np.array(rewards)/1000
plt.plot(opt)

fig = plt.figure(figsize=(8,6))
plt.plot(range(6), eps)
plt.plot(range(2,10), grad)
plt.plot(range(3,10), UCB)
plt.plot(range(5,10), opt)
plt.show()