import numpy as np

class AttentionSequenceReplayBuffer:
    """
    Attention Sequence Replay Buffer
    """
    def __init__(self, num_observations: int, seq_len:int) -> None:
        """
        Constructor
        """
        self.mem_size = 10000
        self.num_observations = num_observations
        self.seq_len = seq_len
        self.mem_ctr = 0
        self.seq_mem_cntr = 0

        #main buffer used for sampling
        self.state_memory = np.zeros((self.mem_size, self.num_observations), dtype = np.float64)
        self.label_memory = np.zeros(self.mem_size, dtype = np.float64)

        #sequence buffer stores 1 sequence of len seq_len, transfers seq to main buffer once full
        self.seq_state_memory = np.zeros((self.seq_len, self.num_observations), dtype=np.float64)

    def store_transition(self, s: np.ndarray, y:float) -> None:
        """
        Store the State and Label 
        """
        mem_index = self.mem_ctr % self.mem_size
        self.seq_state_memory[self.seq_mem_cntr] = s
        self.seq_mem_cntr += 1
        if self.seq_mem_cntr == self.seq_len:
            #Transfer Seq to main mem and clear seq buffer
            for i in range(self.seq_len):
                self.state_memory[mem_index+i] = self.seq_state_memory[i]
            self.label_memory[mem_index] = y
            self.mem_ctr += self.seq_len
            self.seq_mem_cntr = 0

    def get_current_sequence(self) -> list[np.ndarray]:
        """
        get the current sequence of states and labels
        """
        j = self.mem_ctr % self.mem_size
        s = self.state_memory[j-self.seq_len+1:j+1]
        y = self.label_memory[j-self.seq_len+1:j+1]
        return s,y

    def sample_buffer(self, batch_size: int, replace: bool = True) -> list[np.ndarray]:
        """
        get a batch of sequences of states and labels 
        """
        max_mem = min(self.mem_ctr, self.mem_size)
        #selecting starting indices of the sequence in buffer
        indices = [x*self.seq_len for x in range((max_mem//self.seq_len)-1)]
        samples_indices = np.random.choice(indices, batch_size, replace = replace)
        s = np.zeros((batch_size,self.seq_len,self.num_observations))

        for i,j in enumerate(samples_indices):
            s[i] = self.state_memory[j:j+self.seq_len]
        y = self.label_memory[samples_indices] 
        return s, y