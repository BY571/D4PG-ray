
from .networks import DeepActor, Actor, DeepCritic, Critic, IQN, DeepIQN
import numpy as np 
import torch

class Learner():
    def __init__(config, shared_storage, replay_buffer, summary_writer):
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.summary_writer = summary_writer

        # distributional Values
        self.N = 32
        self.entropy_coeff = 0.001
        # munchausen values
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9

        # Actor Network (w/ Target Network)
        if not config.d2rl:
            self.actor_local = Actor(config.state_size, config.action_size, config.seed, hidden_size=config.layer_size).to(config.device)
            self.actor_target = Actor(config.state_size, config.action_size, config.seed, hidden_size=config.layer_size).to(config.device)
        else:
            self.actor_local = DeepActor(config.state_size, config.action_size, config.seed, hidden_size=config.layer_size).to(config.device)
            self.actor_target = DeepActor(config.state_size, config.action_size, config.seed, hidden_size=config.layer_size).to(config.device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_a)

        # Critic Network (w/ Target Network)
        if config.distributional:
            if not config.d2rl:
                self.critic_local = IQN(config.state_size, config.action_size, layer_size=config.layer_size, device=config.device, seed=config.seed, dueling=None, N=self.N).to(config.device)
                self.critic_target = IQN(config.state_size, config.action_size, layer_size=config.layer_size, device=config.device, seed=config.seed, dueling=None, N=self.N).to(config.device)
            else:
                self.critic_local = DeepIQN(config.state_size, config.action_size, layer_size=config.layer_size, device=config.device, seed=config.seed, dueling=None, N=self.N).to(config.device)
                self.critic_target = DeepIQN(config.state_size, config.action_size, layer_size=config.layer_size, device=config.device, seed=config.seed, dueling=None, N=self.N).to(config.device)
        else:
            if not config.d2rl:
                self.critic_local = Critic(config.state_size, config.action_size, config.seed).to(config.device)
                self.critic_target = Critic(config.state_size, config.action_size, config.seed).to(config.device)
            else:
                self.critic_local = DeepCritic(config.state_size, config.action_size, config.seed).to(config.device)
                self.critic_target = DeepCritic(config.state_size, config.action_size, config.seed).to(config.device)

        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_c, weight_decay=0)
    
    
    
    
    def train_network():
    # create critic and target critic 
        while ray.get(replay_buffer.__len__.remote()) == 0:
            pass#
        
        for i in range(self.config.training_steps):

            batch = ray.get(self.replay_buffer.sample.remote()) 
            loss = update_weights(model, batch, optimizer, config)
            soft_update(model, target_model, tau=self.config.tau)

            summary_writer.add_scalar("loss", loss, i)

            ray.get(shared_storage.set_weights.remote(model.to("cpu").state_dict()))
            ray.get(shared_storage.incr_training_counter.remote())
            model.to(config.device)