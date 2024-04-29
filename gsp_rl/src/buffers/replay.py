import numpy as np

class ReplayBuffer():
    """
    SARSD Replay Buffer 
    """
    def __init__(
            self,
            max_size: int,
            num_observations: int,
            num_actions: int,
            action_type: str = None,
    ) -> None:
        """ Constructor """
        self.mem_size = max_size
        self.mem_ctr = 0
        self.action_type = action_type
        self.state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, num_observations), dtype = np.float32)
        if self.action_type == 'Discrete':
            self.action_memory = np.zeros((self.mem_size), dtype = int)
        elif self.action_type == 'Continuous':
            self.action_memory = np.zeros((self.mem_size, num_actions), dtype = np.float32)
        else:
            raise Exception('Unknown Action Type:' + action_type)
        self.reward_memory = np.zeros((self.mem_size), dtype = np.float32)
        self.terminal_memory = np.zeros((self.mem_size), dtype = np.bool_)


    def store_transition(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            state_: np.ndarray,
            done: bool
    ) -> None:
        """
        Stores the SARSD Experience 
        """
        mem_index = self.mem_ctr % self.mem_size
        self.state_memory[mem_index] = state
        self.action_memory[mem_index] = action
        self.reward_memory[mem_index] = reward
        self.new_state_memory[mem_index] = state_
        self.terminal_memory[mem_index] = done
        self.mem_ctr += 1
        

    def sample_buffer(self, batch_size: int) -> list[np.ndarray]:
        """
        Samples the buffer for batch_size experiences"""
        max_mem = min(self.mem_ctr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace = False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones