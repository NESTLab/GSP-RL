import platform
import torch as T


def get_device(recurrent: bool = False) -> T.device:
    """Auto-detect the best available device: cuda > mps > cpu.

    Args:
        recurrent: If True, indicates the network uses LSTM or attention.
                   On macOS MPS, these fall back to CPU due to PyTorch MPS
                   backend bugs with repeated LSTM backward passes.
                   On CUDA (Linux/Windows), recurrent networks use GPU normally.
    """
    if T.cuda.is_available():
        return T.device("cuda:0")
    elif T.backends.mps.is_available():
        if recurrent:
            return T.device("cpu")
        return T.device("mps")
    else:
        return T.device("cpu")


from .dqn import DQN
from .ddqn import DDQN
from .ddpg import DDPGActorNetwork, DDPGCriticNetwork
from .rddpg import RDDPGActorNetwork, RDDPGCriticNetwork
from .td3 import TD3ActorNetwork, TD3CriticNetwork
from .lstm import EnvironmentEncoder
from .self_attention import AttentionEncoder