import gym 
import numpy as np
import torch
import ray
from .networks import Actor, DeepActor
from .utils import OUNoise



@ray.remote
class Worker(object):
    """
    Agent that collects data while interacting with the environment using MCTS. Collected episode trajectories are saved in the replay buffer.
    ============================
    Inputs:
    worker_id  (int)  ID of the worker
    config 
    shared_storage 
    replay buffer     
    """
    def __init__(self, worker_id, config, shared_storage, replay_buffer):
        self.worker_id = worker_id
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        print("init worker: {}".format(self.worker_id))

    def run(self):
        """
        
        """
        print("started worker: {}".format(self.worker_id))

        # build environment
        env = gym.make(self.config.env)
        env.seed(self.worker_id)

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        action_high = env.action_space.high
        action_low = env.action_space.low

        if self.config.noise == "ou":
            noise_func = OUNoise(size=action_size, seed=self.config.seed+self.worker_id)
        else:
            noise_func = None


        # create actor based on config details
        if self.config.d2rl == 1:
            actor = DeepActor(state_size, action_size, noise=noise_func, noise_type=self.config.noise, seed=self.config.seed, hidden_size=self.config.layer_size)
        else:
            actor = Actor(state_size, action_size, noise=noise_func, noise_type=self.config.noise, seed=self.config.seed, hidden_size=self.config.layer_size)

        with torch.no_grad():
            while ray.get(self.shared_storage.get_training_counter.remote()) < self.config.training_steps:
                # get current actor weights
                actor.set_weights(ray.get(self.shared_storage.get_weights.remote()))
                actor.eval()
                state = env.reset()
                done = False
                step = 0
                
                while step <= self.config.max_moves:
                    action = agent.act(state, add_noise=True)
                    action_clipped = np.clip(action*action_high, action_low, action_high)
                    next_state, reward, done, _ = env.step(action_clipped)
                    # add experience to replay buffer
                    self.replay_buffer.add.remote(state, action, reward, next_state, done, worker_id)
                    
                    state = next_state
                    step += 1
                    if done:
                        break
                # add executed steps to step counter
                ray.get(self.shared_storage.incr_interactions.remote(step))
                env.close()
