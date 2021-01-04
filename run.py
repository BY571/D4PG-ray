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
from scripts.training import train

parser = argparse.ArgumentParser(description="")
parser.add_argument("--env", type=str,default="Pendulum-v0", help="Environment name, default = Pendulum-v0")
parser.add_argument("--nstep", type=int, default=1, help ="Nstep bootstrapping, default 1")
parser.add_argument("--per", type=int, default=0, choices=[0,1], help="Adding Priorizied Experience Replay to the agent if set to 1, default = 0")
parser.add_argument("--munchausen", type=int, default=0, choices=[0,1], help="Adding Munchausen RL to the agent if set to 1, default = 0")
parser.add_argument("--iqn", type=int, choices=[0,1], default=0, help="Use distributional IQN Critic if set to 1, default = 1")
parser.add_argument("--noise", type=str, choices=["ou", "gauss"], default="OU", help="Choose noise type: ou = OU-Noise, gauss = Gaussian noise, default ou")
parser.add_argument("--info", type=str, help="Information or name of the run")
parser.add_argument("--device", type=str, default="cpu", help="Training device, default= cpu")
parser.add_argument("--d2rl", type=int, choices=[0,1], default=0, help="Uses Deep Actor and Deep Critic Networks if set to 1 as described in the D2RL Paper: https://arxiv.org/pdf/2010.09163.pdf, default=0")
parser.add_argument("--frames", type=int, default=30000, help="The amount of training interactions with the environment, default is 100000")
parser.add_argument("--training_steps", type=int, default=3000, help="Numnber of backprop steps, default=10000")
parser.add_argument("--seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("--lr_a", type=float, default=5e-4, help="Actor learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("--lr_c", type=float, default=5e-4, help="Critic learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("--layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-3") #for per 1e-2 for regular 1e-3 -> Pendulum!
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("--worker_number", type=int, default=1, help="Number of parallel Worker to gather experience, default = 4")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="Number of Network Updates befor next Evaluation run, default 10")
args = parser.parse_args()

def timer(start,end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

if __name__ == "__main__":
    ray.init()
    writer = SummaryWriter("runs/"+args.info)
    # if training
    #take time
    t0 = time.time()
    trained_model = train(args, writer)
    t1 = time.time()
    time.sleep(1.5)
    timer(t0, t1)
    # else:
        # load_weights
        # evaluate

    # if save model
    # save_weights