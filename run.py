import ray
import numpy as np
import random
import gym
from collections import deque
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
from scripts.agent import Agent
import json
from scripts import training


parser = argparse.ArgumentParser(description="")
parser.add_argument("--env", type=str,default="Pendulum-v0", help="Environment name, default = Pendulum-v0")
parser.add_argument("--nstep", type=int, default=1, help ="Nstep bootstrapping, default 1")
parser.add_argument("--per", type=int, default=0, choices=[0,1], help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")
parser.add_argument("--munchausen", type=int, default=0, choices=[0,1], help="Adding Munchausen RL to the agent if set to 1, default = 0")
parser.add_argument("--iqn", type=int, choices=[0,1], default=0, help="Use distributional IQN Critic if set to 1, default = 1")
parser.add_argument("--noise", type=str, choices=["ou", "gauss"], default="OU", help="Choose noise type: ou = OU-Noise, gauss = Gaussian noise, default ou")
parser.add_argument("--info", type=str, help="Information or name of the run")
parser.add_argument("--d2rl", type=int, choices=[0,1], default=0, help="Uses Deep Actor and Deep Critic Networks if set to 1 as described in the D2RL Paper: https://arxiv.org/pdf/2010.09163.pdf, default=0")
parser.add_argument("--frames", type=int, default=20000, help="The amount of training interactions with the environment, default is 100000")
parser.add_argument("--eval_every", type=int, default=1000, help="Number of interactions after which the evaluation runs are performed, default = 1000")
parser.add_argument("--eval_runs", type=int, default=1, help="Number of evaluation runs performed, default = 1")
parser.add_argument("--seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("--lr_a", type=float, default=5e-4, help="Actor learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("--lr_c", type=float, default=5e-4, help="Critic learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("--learn_every", type=int, default=1, help="Learn every x interactions, default = 1")
parser.add_argument("--learn_number", type=int, default=1, help="Learn x times per interaction, default = 1")
parser.add_argument("--layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-3") #for per 1e-2 for regular 1e-3 -> Pendulum!
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("--worker_number", type=int, default=4, help="Number of parallel Worker to gather experience, default = 4")
args = parser.parse_args()


if __name__ == "__main__":
    writer = SummaryWriter("runs/"+args.info)

    # if training
    trained_model = training(args, writer)
    # else:
        # load_weights
        # evaluate

    # if save model
    # save_weights