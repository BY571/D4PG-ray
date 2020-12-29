
import ray
import random 
import torch
import numpy as np
from collections import namedtuple, deque

@ray.remote
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, config):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            n_step (int): bootstrapping steps
            device (str): training device
            gamma (float): bootstrapping gamma
        """
        self.device = config.device
        self.memory = deque(maxlen=config.replay_memory)  
        self.batch_size = config.batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(config.seed)
        self.gamma = config.gamma
        self.n_step_buffer = {str(w): deque(maxlen=config.n_step) for w in range(config.worker_number)} 
        self.iter_ = 0
        self.n_step = config.n_step

    def add(self, state, action, reward, next_state, done, worker_number):
        """Add a new experience to memory."""

        self.n_step_buffer[str(worker_number)].append((state, action, reward, next_state, done))
        if len(self.n_step_buffer[str(worker_number)]) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer[str(worker_number)])
            e = self.experience(state, action, reward, next_state, done)
            self.memory.append(e)


    def calc_multistep_return(self, n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, None, None)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
@ray.remote
class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, batch_size, device, seed, gamma=0.99, n_step=1, alpha=0.6, beta_start = 0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.device = device
        self.frame = 1 #for beta calculation
        self.batch_size = batch_size
        self.capacity   = capacity
        self.buffer     = deque(maxlen=capacity)
        self.pos        = 0
        self.priorities = deque(maxlen=capacity)
        self.seed = np.random.seed(seed)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def calc_multistep_return(self,n_step_buffer):
        Return = 0
        for idx in range(self.n_step):
            Return += self.gamma**idx * n_step_buffer[idx][2]
        
        return n_step_buffer[0][0], n_step_buffer[0][1], Return, n_step_buffer[-1][3], n_step_buffer[-1][4]

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        action = torch.from_numpy(action).unsqueeze(0)

        # n_step calc
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_step:
            state, action, reward, next_state, done = self.calc_multistep_return(self.n_step_buffer)

        max_prio = np.array(self.priorities, dtype=float).max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        

        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)


        
    def sample(self):
        N = len(self.buffer)
        prios = np.array(self.priorities, dtype=float)
        assert N == len(prios)
            
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p
        indices = np.random.choice(N, self.batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        #print(beta)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (N * P[indices])**(-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 

        states      = torch.FloatTensor(np.float32(np.concatenate(states))).to(self.device)
        next_states = torch.FloatTensor(np.float32(np.concatenate(next_states))).to(self.device)
        actions     = torch.cat(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones       = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights    = torch.FloatTensor(weights).unsqueeze(1)
        #print("s",states.shape)
        #print("ns", next_states.shape)
        #print("a", actions.shape)
        #print("r", rewards.shape)
        #print("d", dones.shape)
        #print("w", weights.shape)
        
        return states, actions, rewards, next_states, dones, indices, weights
    

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio 

    def __len__(self):
        return len(self.buffer)