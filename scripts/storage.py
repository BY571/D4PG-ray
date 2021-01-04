import ray
import gym
from .networks import Actor, DeepActor  


@ray.remote
class SharedStorage(object):
    """
    Storage that keeps the current weights of the model and counts the interactions of the worker with the environment.
    Input: 
    model

    TODO: 
    ADD metrics that I want to track
    """
    def __init__(self, config):
        self.update_counter = 0
        self.interaction_counter = 0
        self.config = config

        # create actor based on config details
        if config.d2rl == 1:
            self.actor = DeepActor(config.state_size, config.action_size, noise=None, noise_type=self.config.noise, seed=self.config.seed, hidden_size=config.layer_size)
        else:
            self.actor = Actor(config.state_size, config.action_size, noise=None, noise_type=self.config.noise, seed=self.config.seed, hidden_size=config.layer_size)
        self.evaluation_reward_history = {} # key: learning step, value: reward

    def get_weights(self):
        return self.actor.get_weights()

    def set_weights(self, weights):
        return self.actor.set_weights(weights)

    def increase_update_coutner(self):
        self.update_counter += 1

    def get_training_counter(self):
        return self.update_counter

    def set_eval_reward(self, step, rewards):
        self.evaluation_reward_history[step] = rewards

    def get_latest_reward(self):
        return list(self.evaluation_reward_history.keys())[-1]

    def incr_interactions(self, steps):
        self.interaction_counter += steps
    
    def get_interactions(self):
        return self.interaction_counter