#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:41:23 2017

@author: levimcclenny
"""
import numpy as np

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
            optimalAction = []
            for j in range(0,time):
                optimal_count = 0
                if eps[k] > np.random.random(): 
                    action = np.random.randint(arms)
                else:
                    action = np.argmax(q_est)
                if UCB:
                    action = np.argmax(q_est + c*np.sqrt(np.log(time+1)/(counts+1)))
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

rewardOpt, actionOpt = k_bandit_testbed(10, 10000, 2000, [0], constant_step = True, 
                                          step_size = 0.05, optimistic = True)
rewardOpt2, actionOpt2 = k_bandit_testbed(10, 10000, 2000, [0], constant_step = True, 
                                          step_size = 0.025, optimistic = True)
rewardOpt3, actionOpt3 = k_bandit_testbed(10, 10000, 2000, [0], constant_step = True, 
                                          step_size = 0.005, optimistic = True)

fig = plt.subplots(figsize=(14, 10))
plt.plot(actionOpt[0], label = "Optimistic/Greedy, $Q_{1} = 5$, step size = 0.1")
plt.plot(actionOpt2[0], label = "Optimistic/Greedy, $Q_{1} = 5$, step size = 0.025")
plt.plot(actionOpt3[0], label = "Optimistic/Greedy, $Q_{1} = 5$, step size = 0.005")
plt.xlabel('Steps')
plt.ylabel('% optimal action')
plt.legend()