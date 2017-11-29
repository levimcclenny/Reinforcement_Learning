#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:45:41 2017

@author: levimcclenny
"""
import numpy as np
    

gridsize = 4
    
def movestate(inputstate, nextstate):
    if inputstate == [0,0]:
        reward = 0
        prob = .25
        outstate = [0,0]
    elif inputstate == [gridsize-1, gridsize-1]:
        reward = 0
        prob = .25
        outstate = [gridsize-1, gridsize-1]
    elif nextstate[0] > gridsize -1 or nextstate[0] < 0 or nextstate[1] > gridsize -1 or nextstate[1] < 0:
        outstate = inputstate
        prob = .25
        reward = -1
    else:
        outstate = nextstate
        prob = .25
        reward = -1
    return outstate, prob, reward


grid = np.zeros((gridsize, gridsize))   

for k in range(1000):
    
    newgrid = np.zeros((gridsize, gridsize))  
    if k == 0:
        print('k = 0')
        print(newgrid)
    for i in range(gridsize):
        for j in range(gridsize):
            Outstate, Prob, Reward = movestate([i,j], [i+1, j])
            newgrid[i, j] += Prob * (Reward + grid[Outstate[0], Outstate[1]])
            Outstate, Prob, Reward = movestate([i,j], [i-1, j])
            newgrid[i, j] += Prob * (Reward + grid[Outstate[0], Outstate[1]])
            Outstate, Prob, Reward = movestate([i,j], [i, j+1])
            newgrid[i, j] += Prob * (Reward + grid[Outstate[0], Outstate[1]])
            Outstate, Prob, Reward = movestate([i,j], [i, j-1])
            newgrid[i, j] += Prob * (Reward + grid[Outstate[0], Outstate[1]])
    if k == 0 or k == 1 or k == 2 or k == 9:
        print("k = ",k+1)
        print(np.round(newgrid,1))
    elif np.sum(np.abs(grid - newgrid)) < 1e-4:
        print('k = inf')
        print(np.round(newgrid,1))
        break
    grid = newgrid


grid = np.zeros((gridsize, gridsize))     
for k in range(1000):
    newgrid = np.zeros((gridsize, gridsize))  
    if k == 0:
        print('k = 0')
        print(newgrid)  

    for i in range(gridsize):
        for j in range(gridsize):
            value = []
            
            Outstate, Prob, Reward = movestate([i,j], [i+1, j])
            value.append(Reward + grid[Outstate[0], Outstate[1]])
            
            Outstate, Prob, Reward = movestate([i,j], [i-1, j])
            value.append(Reward + grid[Outstate[0], Outstate[1]])
            
            Outstate, Prob, Reward = movestate([i,j], [i, j+1])
            value.append(Reward + grid[Outstate[0], Outstate[1]])
            
            Outstate, Prob, Reward = movestate([i,j], [i, j-1])
            value.append(Reward + grid[Outstate[0], Outstate[1]])
            
            newgrid[i,j] = np.max(value)
            
    if k == 0 or k == 1 or k == 2:
        print("k = ",k+1)
        print(np.round(newgrid,1))
    elif np.sum(np.abs(grid - newgrid)) < 1e-4:
        print('k = inf')
        print(np.round(newgrid,1))
        break
    grid = newgrid

    