# nn-planning
Julia codes for model-free scale-free automated planning using deep neural networks.

This repository contains codes for expansion network (scale-free, model-free framework for training state-transition function in automated planning) and 
several architectures of heuristic networks (scale-free, model-free frameworks for training heuristic function for state-space search). There are three heuristic networks: 
- CNN (convolutional network)
- CNN_att (convolutional network using soft attention masks) 
- RNN (recurrent network based on MAC reasoning recurrent cell with self-stopping) 

All networks can be trained for arbitrary problem. At the moment we use 4 domains - Single-agent maze, Multi-goal maze, Multi-agent maze, Sokoban. Datasets used to train the networks 
are too large to store in Git repository, contact me for access to the generated data. Maze data was created using modified Prim's algorithm, Sokoban data is generated from 3x3 templates.

In folder planning_experiments_mac are last performed planning experiments on planning data (50 unseen instances per domain) with classical planning heuristics (blind, Euclidean distance, LM-cut,
H^FF) and the three types of heuristic networks. There are 4 data sizes for all maze domains - 8x8, 16x16, 32x32, 64x64 and three sizes for Sokoban - 8x8, 16x16, 10x10 which are from the 
Boxoban dataset. All experiments can be run with the provided solvers + expansion network for three maze domains is in the exp_nets folder (saved as parameters due to previous Julia compatibility problems). 

With any questions contact me at michaela.urbanovska@fel.cvut.cz 

