import numpy as np

class SequenceReplayBuffer:
    """
    Sequence Replay Buffer
    """
    def __init__(
            self,
            max_sequence: int,
            num_observations: int,
            num_actions: int,
            seq_len: int
    ) -> None:
        """
        Constructor 
        """
        self.mem_size = max_sequence*seq_len
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.seq_len = seq_len
        self.mem_ctr = 0
        self.seq_mem_cntr = 0

        #main buffer used for sampling
        self.state_memory = np.zeros((self.mem_size, self.num_observations), dtype = np.float64)
        self.action_memory = np.zeros((self.mem_size, self.num_actions), dtype = np.float64)
        self.new_state_memory = np.zeros((self.mem_size, self.num_observations), dtype = np.float64)
        self.reward_memory = np.zeros((self.mem_size), dtype = np.float64)
        self.terminal_memory = np.zeros((self.mem_size), dtype = np.bool)

        #sequence buffer stores 1 sequence of len seq_len, transfers seq to main buffer once full
        self.seq_state_memory = np.zeros((self.seq_len, self.num_observations), dtype=np.float64)
        self.seq_action_memory = np.zeros((self.seq_len, self.num_actions), dtype=np.float64)
        self.seq_new_state_memory = np.zeros((self.seq_len, self.num_observations), dtype = np.float64)
        self.seq_reward_memory = np.zeros((self.seq_len), dtype = np.float64)
        self.seq_terminal_memory = np.zeros((self.seq_len), dtype = np.bool)

    def store_transition(
            self,
            s: np.ndarray,
            a: np.ndarray,
            r: float,
            s_: np.ndarray,
            d: bool
    ) -> None:
        """
        Store the SARSD Experience until the trajectory length is met
        """
        mem_index = self.mem_ctr % self.mem_size
        # import ipdb; ipdb.set_trace()
        self.seq_state_memory[self.seq_mem_cntr] = s
        self.seq_action_memory[self.seq_mem_cntr] = a
        self.seq_new_state_memory[self.seq_mem_cntr] = s_
        self.seq_reward_memory[self.seq_mem_cntr] = r
        self.seq_terminal_memory[self.seq_mem_cntr] = d
        self.seq_mem_cntr += 1
        
        if self.seq_mem_cntr == self.seq_len:
            #Transfer Seq to main mem and clear seq buffer
            for i in range(self.seq_len):
                self.state_memory[mem_index+i] = self.seq_state_memory[i]
                self.action_memory[mem_index+i] = self.seq_action_memory[i]
                self.new_state_memory[mem_index+i] = self.seq_new_state_memory[i]
                self.reward_memory[mem_index+i] = self.seq_reward_memory[i]
                self.terminal_memory[mem_index+i] = self.seq_terminal_memory[i]
            self.mem_ctr += self.seq_len
            self.seq_mem_cntr = 0

    def get_current_sequence(self) -> list[np.ndarray]:
        """
        get the current trajectory
        """
        j = self.mem_ctr % self.mem_size
        s = self.state_memory[j:j+self.seq_len]
        s_ = self.new_state_memory[j:j+self.seq_len]
        a = self.action_memory[j:j+self.seq_len]
        r = self.reward_memory[j:j+self.seq_len]
        d = self.terminal_memory[j:j+self.seq_len]
        return s,s_,a,r,d

    def sample_buffer(self, batch_size: int, replace: bool = True) -> list[np.ndarray]:
        """
        Sample the buffer for batch_size sequences of SARSD experiences
        """
        max_mem = min(self.mem_ctr, self.mem_size)
        #selecting starting indices of the sequence in buffer
        indices = [x*self.seq_len for x in range((max_mem//self.seq_len)-1)]
        samples_indices = np.random.choice(indices, batch_size, replace = replace)
        s = np.zeros((batch_size,self.seq_len,self.num_observations))
        s_ = np.zeros((batch_size,self.seq_len,self.num_observations))
        a = np.zeros((batch_size,self.seq_len,self.num_actions))
        r = np.zeros((batch_size, self.seq_len), dtype= np.float64)
        d = np.zeros((batch_size, self.seq_len), dtype= np.bool)
        for i,j in enumerate(samples_indices):
            s[i] = self.state_memory[j:j+self.seq_len]
            s_[i] = self.new_state_memory[j:j+self.seq_len]
            a[i] = self.action_memory[j:j+self.seq_len]
            r[i] = self.reward_memory[j:j+self.seq_len]
            d[i] = self.terminal_memory[j:j+self.seq_len]
        return s, a, r, s_, d