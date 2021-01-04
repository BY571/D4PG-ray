
from .networks import DeepActor, Actor, DeepCritic, Critic, IQN, DeepIQN
import numpy as np 
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import ray

class Learner():
    def __init__(self, config, shared_storage, replay_buffer, summary_writer):
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.summary_writer = summary_writer
        self.device = config.device
        self.gamma = config.gamma
        self.n_step = config.nstep
        self.BATCH_SIZE = config.batch_size
        self.per = config.per
        self.TAU = config.tau

        # distributional Values
        self.N = 32
        self.entropy_coeff = 0.001
        # munchausen values
        self.munchausen = config.munchausen
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9


        # create critic and target critic 
        # Actor Network (w/ Target Network)
        if not config.d2rl:
            self.actor_local = Actor(config.state_size, config.action_size, noise=None, noise_type="gauss", seed=config.seed, hidden_size=config.layer_size).to(config.device)
            self.actor_target = Actor(config.state_size, config.action_size, noise=None, noise_type="gauss", seed=config.seed, hidden_size=config.layer_size).to(config.device)
        else:
            self.actor_local = DeepActor(config.state_size, config.action_size, noise=None, noise_type="gauss", seed=config.seed, hidden_size=config.layer_size).to(config.device)
            self.actor_target = DeepActor(config.state_size, config.action_size, noise=None, noise_type="gauss", seed=config.seed, hidden_size=config.layer_size).to(config.device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_a)

        # Critic Network (w/ Target Network)
        if config.iqn:
            self.update_weights = self.learn_distribution
            if not config.d2rl:
                self.critic_local = IQN(config.state_size, config.action_size, layer_size=config.layer_size, device=config.device, seed=config.seed, dueling=None, N=self.N).to(config.device)
                self.critic_target = IQN(config.state_size, config.action_size, layer_size=config.layer_size, device=config.device, seed=config.seed, dueling=None, N=self.N).to(config.device)
            else:
                self.critic_local = DeepIQN(config.state_size, config.action_size, layer_size=config.layer_size, device=config.device, seed=config.seed, dueling=None, N=self.N).to(config.device)
                self.critic_target = DeepIQN(config.state_size, config.action_size, layer_size=config.layer_size, device=config.device, seed=config.seed, dueling=None, N=self.N).to(config.device)
        else:
            self.update_weights = self.learn_
            if not config.d2rl:
                self.critic_local = Critic(config.state_size, config.action_size, config.seed).to(config.device)
                self.critic_target = Critic(config.state_size, config.action_size, config.seed).to(config.device)
            else:
                self.critic_local = DeepCritic(config.state_size, config.action_size, config.seed).to(config.device)
                self.critic_target = DeepCritic(config.state_size, config.action_size, config.seed).to(config.device)

        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_c, weight_decay=0)

    
    
    def train_network(self):

        while ray.get(self.replay_buffer.__len__.remote()) < self.BATCH_SIZE:
            pass
            
        for i in range(self.config.training_steps):

            batch = ray.get(self.replay_buffer.sample.remote()) 
            c_loss, a_loss = self.update_weights(batch)

            self.summary_writer.add_scalar("Critic loss", c_loss, i)
            self.summary_writer.add_scalar("Actor loss", a_loss, i)
            steps = ray.get(self.shared_storage.get_interactions.remote())
            self.summary_writer.add_scalar("Evaluation Reward", 
                                ray.get(self.shared_storage.get_latest_reward.remote()), 
                                steps)

            ray.get(self.shared_storage.set_weights.remote(self.actor_local.to("cpu").state_dict()))
            ray.get(self.shared_storage.increase_update_coutner.remote())
            self.actor_local.to(self.config.device)




    # different learn functions 
    def learn_(self, experiences):
            """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            states, actions, rewards, next_states, dones, idx, weights = experiences

            # ---------------------------- update critic ---------------------------- #
            if not self.munchausen:
                # Get predicted next-state actions and Q values from target models
                with torch.no_grad():
                    actions_next = self.actor_target(next_states.to(self.device))
                    Q_targets_next = self.critic_target(next_states.to(self.device), actions_next.to(self.device))
                    # Compute Q targets for current states (y_i)
                    Q_targets = rewards + (self.gamma**self.n_step * Q_targets_next * (1 - dones))
            else:
                with torch.no_grad():
                    actions_next = self.actor_target(next_states.to(self.device))
                    q_t_n = self.critic_target(next_states.to(self.device), actions_next.to(self.device))
                    # calculate log-pi - in the paper they subtracted the max_Q value from the Q to ensure stability since we only predict the max value we dont do that
                    # this might cause some instability (?) needs to be tested
                    logsum = torch.logsumexp(\
                        q_t_n /self.entropy_tau, 1).unsqueeze(-1) #logsum trick
                    assert logsum.shape == (self.BATCH_SIZE, 1), "log pi next has wrong shape: {}".format(logsum.shape)
                    tau_log_pi_next = (q_t_n  - self.entropy_tau*logsum)
                    
                    pi_target = F.softmax(q_t_n/self.entropy_tau, dim=1)
                    # in the original paper for munchausen RL they summed over all actions - we only predict the best Qvalue so we will not sum over all actions
                    Q_target = (self.gamma**self.n_step * (pi_target * (q_t_n-tau_log_pi_next)*(1 - dones)))
                    assert Q_target.shape == (self.BATCH_SIZE, 1), "has shape: {}".format(Q_target.shape)

                    q_k_target = self.critic_target(states, actions)
                    tau_log_pik = q_k_target - self.entropy_tau*torch.logsumexp(\
                                                                            q_k_target/self.entropy_tau, 1).unsqueeze(-1)
                    assert tau_log_pik.shape == (self.BATCH_SIZE, 1), "shape instead is {}".format(tau_log_pik.shape)
                    # calc munchausen reward:
                    munchausen_reward = (rewards + self.alpha*torch.clamp(tau_log_pik, min=self.lo, max=0))
                    assert munchausen_reward.shape == (self.BATCH_SIZE, 1)
                    # Compute Q targets for current states 
                    Q_targets = munchausen_reward + Q_target
            # Compute critic loss
            Q_expected = self.critic_local(states, actions)
            if self.per:
                td_error =  Q_targets - Q_expected
                critic_loss = (td_error.pow(2)*weights.to(self.device)).mean().to(self.device)
            else:
                critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(self.critic_local.parameters(), 1)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.actor_local, self.actor_target)                     
            if self.per:
                self.memory.update_priorities(idx, np.clip(abs(td_error.data.cpu().numpy()),-1,1))
            # ----------------------- update epsilon and noise ----------------------- #
            
            #self.epsilon *= self.EPSILON_DECAY
            
            return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()
                
    def learn_distribution(self, experiences):
                """Update policy and value parameters using given batch of experience tuples.
                Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
                where:
                    actor_target(state) -> action
                    critic_target(state, action) -> Q-value
                Params
                ======
                    experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                    gamma (float): discount factor
                """
                states, actions, rewards, next_states, dones, idx, weights = experiences

                # ---------------------------- update critic ---------------------------- #
                # Get predicted next-state actions and Q values from target models

                # Get max predicted Q values (for next states) from target model
                if not self.munchausen:
                    with torch.no_grad():
                        next_actions = self.actor_local(next_states)
                        Q_targets_next, _ = self.critic_target(next_states, next_actions, self.N)
                        Q_targets_next = Q_targets_next.transpose(1,2)
                    # Compute Q targets for current states 
                    Q_targets = rewards.unsqueeze(-1) + (self.gamma**self.n_step * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
                else:
                    with torch.no_grad():
                        #### CHECK FOR THE SHAPES!!
                        actions_next = self.actor_target(next_states.to(self.device))
                        Q_targets_next, _ = self.critic_target(next_states.to(self.device), actions_next.to(self.device), self.N)

                        q_t_n = Q_targets_next.mean(1)
                        # calculate log-pi - in the paper they subtracted the max_Q value from the Q to ensure stability since we only predict the max value we dont do that
                        # this might cause some instability (?) needs to be tested
                        logsum = torch.logsumexp(\
                            q_t_n /self.entropy_tau, 1).unsqueeze(-1) #logsum trick
                        assert logsum.shape == (self.BATCH_SIZE, 1), "log pi next has wrong shape: {}".format(logsum.shape)
                        tau_log_pi_next = (q_t_n  - self.entropy_tau*logsum).unsqueeze(1)
                        
                        pi_target = F.softmax(q_t_n/self.entropy_tau, dim=1).unsqueeze(1)
                        # in the original paper for munchausen RL they summed over all actions - we only predict the best Qvalue so we will not sum over all actions
                        Q_target = (self.gamma**self.n_step * (pi_target * (Q_targets_next-tau_log_pi_next)*(1 - dones.unsqueeze(-1)))).transpose(1,2)
                        assert Q_target.shape == (self.BATCH_SIZE, self.action_size, self.N), "has shape: {}".format(Q_target.shape)

                        q_k_target = self.critic_target.get_qvalues(states, actions)
                        tau_log_pik = q_k_target - self.entropy_tau*torch.logsumexp(\
                                                                                q_k_target/self.entropy_tau, 1).unsqueeze(-1)
                        assert tau_log_pik.shape == (self.BATCH_SIZE, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
                        # calc munchausen reward:
                        munchausen_reward = (rewards + self.alpha*torch.clamp(tau_log_pik, min=self.lo, max=0)).unsqueeze(-1)
                        assert munchausen_reward.shape == (self.BATCH_SIZE, self.action_size, 1)
                        # Compute Q targets for current states 
                        Q_targets = munchausen_reward + Q_target
                # Get expected Q values from local model
                Q_expected, taus = self.critic_local(states, actions, self.N)
                assert Q_targets.shape == (self.BATCH_SIZE, 1, self.N)
                assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)
        
                # Quantile Huber loss
                td_error = Q_targets - Q_expected
                assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
                huber_l = calculate_huber_loss(td_error, 1.0)
                quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
                
                if self.per:
                    critic_loss = (quantil_l.sum(dim=1).mean(dim=1, keepdim=True)*weights.to(self.device)).mean()
                else:
                    critic_loss = quantil_l.sum(dim=1).mean(dim=1).mean()
                # Minimize the loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                clip_grad_norm_(self.critic_local.parameters(), 1)
                self.critic_optimizer.step()

                # ---------------------------- update actor ---------------------------- #
                # Compute actor loss
                actions_pred = self.actor_local(states)
                actor_loss = -self.critic_local.get_qvalues(states, actions_pred).mean()
                # Minimize the loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ----------------------- update target networks ----------------------- #
                self.soft_update(self.critic_local, self.critic_target)
                self.soft_update(self.actor_local, self.actor_target)                     
                if self.per:
                    self.memory.update_priorities(idx, np.clip(abs(td_error.sum(dim=1).mean(dim=1,keepdim=True).data.cpu().numpy()),-1,1))
                # ----------------------- update epsilon and noise ----------------------- #
                
                return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()

        
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)
