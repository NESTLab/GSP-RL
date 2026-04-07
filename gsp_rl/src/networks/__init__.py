import torch as T


def get_device() -> T.device:
    """Auto-detect the best available device: cuda > mps > cpu."""
    if T.cuda.is_available():
        return T.device("cuda:0")
    elif T.backends.mps.is_available():
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