#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:45:41 2017

@author: levimcclenny
"""
import numpy as np
    
    
def movestate(inputstate, nextstate):
    if inputstate == [0,1]:
        reward = 10
        prob = .25
        outstate = [4,1]
    elif inputstate == [0,3]:
        reward = 5
        prob = .25
        outstate = [2,3]
    elif nextstate[0] > 4 or nextstate[0] < 0 or nextstate[1] > 4 or nextstate[1] < 0:
        outstate = inputstate
        prob = .25
        reward = -1
    else:
        outstate = nextstate
        prob = .25
        reward = 0
    return outstate, prob, reward


grid = np.zeros((5,5))   

while True:
    
    newgrid = np.zeros((5,5))  

    for i in range(5):
        for j in range(5):
            Outstate, Prob, Reward = movestate([i,j], [i+1, j])
            newgrid[i, j] += Prob * (Reward + .9 * grid[Outstate[0], Outstate[1]])
            Outstate, Prob, Reward = movestate([i,j], [i-1, j])
            newgrid[i, j] += Prob * (Reward + .9 * grid[Outstate[0], Outstate[1]])
            Outstate, Prob, Reward = movestate([i,j], [i, j+1])
            newgrid[i, j] += Prob * (Reward + .9 * grid[Outstate[0], Outstate[1]])
            Outstate, Prob, Reward = movestate([i,j], [i, j-1])
            newgrid[i, j] += Prob * (Reward + .9 * grid[Outstate[0], Outstate[1]])
    if np.sum(np.abs(grid - newgrid)) < 1e-4:
        print('Random Policy')
        print(np.round(newgrid,1))
        break
    grid = newgrid




grid = np.zeros((5,5))   

while True:
    
    newgrid = np.zeros((5,5))  

    for i in range(5):
        for j in range(5):
            value = []
            
            Outstate, Prob, Reward = movestate([i,j], [i+1, j])
            value.append(Reward + .9 * grid[Outstate[0], Outstate[1]])
            
            Outstate, Prob, Reward = movestate([i,j], [i-1, j])
            value.append(Reward + .9 * grid[Outstate[0], Outstate[1]])
            
            Outstate, Prob, Reward = movestate([i,j], [i, j+1])
            value.append(Reward + .9 * grid[Outstate[0], Outstate[1]])
            
            Outstate, Prob, Reward = movestate([i,j], [i, j-1])
            value.append(Reward + .9 * grid[Outstate[0], Outstate[1]])
            
            newgrid[i,j] = np.max(value)
            
    if np.sum(np.abs(grid - newgrid)) < 1e-4:
        print('Optimal Value Policy')
        print(np.round(newgrid, 1))
        break
    grid = newgrid


    