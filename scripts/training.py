import torch
import numpy as np
from .storage import SharedStorage
from .replay_buffer import ReplayBuffer, PrioritizedReplay
import gym
from .networks import DeepActor, Actor, DeepCritic, Critic, IQN, DeepIQN
from .learner import Learner
from .worker import Worker
import ray






@ray.remote
def evaluation(config, shared_storage):
    """
    Runs the evaluation runs every x training steps and add the results to the replay buffer

    """
    # 1 init model & load weights
    # 2 init new game 
    # 3 run evaluation once number of interactions got reached
    # 4 save results on storage  

    # build environment
    env = gym.make(config.env)
    env.seed(config.seed)
    action_high = config.action_high
    action_low = config.action_low


    # create actor based on config details
    if config.d2rl == 1:
        actor = DeepActor(config.state_size, config.action_size, noise=None, noise_type=config.noise, seed=config.seed, hidden_size=config.layer_size)
    else:
        actor = Actor(config.state_size, config.action_size, noise=None, noise_type=config.noise, seed=config.seed, hidden_size=config.layer_size)

    actor.eval()

    with torch.no_grad():
        while ray.get(shared_storage.get_training_counter.remote()) < config.training_steps:
            if ray.get(shared_storage.get_training_counter.remote()) % config.checkpoint_interval == 0:
                # run eval game
                actor.set_weights(ray.get(shared_storage.get_weights.remote()))
                counter = ray.get(shared_storage.get_interactions.remote())
                state = env.reset()
                done = False
                rewards = 0
                while not done:
                    action = actor.act(state)
                    action_clipped = np.clip(action*action_high, action_low, action_high)
                    state, reward, done, _ = env.step(action_clipped)
                    rewards += reward
                    if done:
                        break
                env.close()
                print("Actor Steps: {} | Evaluation Rewards: {} | Learning Steps: {} ".format(counter, rewards, ray.get(shared_storage.get_training_counter.remote())))
                shared_storage.set_eval_reward.remote(counter, rewards)


        



def train(config, summary_writer):
    """
    Trains the Agent. Distributed Worker collect data and store it in the replay buffer. On certain update steps the weights of the Networks get optimized. 

    Returns the trained model.
    =====
    Inputs

    config: Agent configuration
    summary writer: 

    """
    # build environment
    env = gym.make(config.env)
    config.state_size = env.observation_space.shape[0]
    config.action_size = env.action_space.shape[0]
    config.action_high = env.action_space.high
    config.action_low = env.action_space.low

    config.batch_size = config.worker_number*config.batch_size


    # initialize storage and replay buffer 
    storage = SharedStorage.remote(config)
    if config.per == 0:
        replay_buffer = ReplayBuffer.remote(config=config)
    else:
        replay_buffer = ReplayBuffer.remote(config=config)

    # create a number of distributed worker 
    workers = [Worker.remote(worker_id, config, storage, replay_buffer) for worker_id in range(0, config.worker_number)]
    # collect experience
    for w in workers: w.run.remote()
    # add evaluation worker 
    evals =  [evaluation.remote(config, storage)]
    # Create Learner
    learner = Learner(config, storage, replay_buffer, summary_writer)
    # learn
    learner.train_network()

    ray.get(evals)
    return ray.get(storage.get_weights.remote())