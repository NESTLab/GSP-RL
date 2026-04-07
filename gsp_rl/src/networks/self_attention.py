"""Attention-based encoder for A-GSP (Attention Global State Prediction).

Provides SelfAttention (multi-head), TransformerBlock (attention + feedforward
with skip connections), and AttentionEncoder (full encoder with positional
embeddings that outputs bounded predictions). Used for supervised global
state prediction from observation sequences.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gsp_rl.src.networks import get_device


class SelfAttention(nn.Module):
    """Multi-head self-attention layer.

    Splits embed_size into `heads` parallel attention heads of size
    head_dim = embed_size // heads. Uses Einstein summation for the
    attention computation.

    Attributes:
        head_dim: Dimension per attention head.
        values, keys, query: Linear projections (head_dim -> head_dim, no bias).
        fc_out: Output projection (embed_size -> embed_size).
    """
    def __init__(self, embed_size: int, heads: int) -> None:
        """Initialize multi-head self-attention.

        Args:
            embed_size: Total embedding dimension (must be divisible by heads).
            heads: Number of parallel attention heads.
        """
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed Size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(self.heads*self.head_dim, embed_size)

        self.softmax = nn.Softmax(dim = 3)

    def forward(
            self,
            values: T.Tensor,
            keys: T.Tensor,
            query: T.Tensor,
            mask: bool = None
    ) -> T.Tensor:
        """Compute multi-head self-attention.

        Args:
            values: Shape (N, value_len, embed_size).
            keys: Shape (N, key_len, embed_size).
            query: Shape (N, query_len, embed_size).
            mask: Optional attention mask (for decoder use).

        Returns:
            Attention output of shape (N, query_len, embed_size).
        """
        # mask is only for the decoder
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = T.einsum("nqhd, nkhd->nhqk", [queries, keys])
        # query shape = N, query_len, heads, heads_dim
        # key shape = N, key_len, heads, heads_dim
        # energy shape = N, heads, query_len, key_len
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        attention = self.softmax(energy / (self.embed_size**(1/2)))

        out = T.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape = N, heads, querry_len, key_len
        # values shape = N, value_len, heads, heads_dim
        # out = N, query_len, heads, heads_dim, then flatten last two dims

        return self.fc_out(out)

############################################################################
# Transformer
############################################################################
class TransformerBlock(nn.Module):
    """Single transformer encoder block with self-attention and feedforward.

    Structure: SelfAttention -> LayerNorm (skip) -> FeedForward -> LayerNorm (skip).
    Both skip connections use dropout.
    """
    def __init__(self, embed_size: int, heads: int, dropout: float, forward_expansion: float):
        """Initialize transformer block.

        Args:
            embed_size: Embedding dimension.
            heads: Number of attention heads.
            dropout: Dropout probability.
            forward_expansion: Feedforward hidden layer multiplier.
        """
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            value: T.Tensor,
            key: T.Tensor,
            query: T.Tensor,
            mask: bool = None
    ) -> T.Tensor:
        """ Forward Propogation Step """
        # mask is only for the decoder
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query)) # skip connection in encoder block
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward+x)) # skip connection in encoder block
        return out

############################################################################
# Attention Encoder
############################################################################
class AttentionEncoder(nn.Module):
    """Attention-based encoder for A-GSP global state prediction.

    Processes a sequence of observations into a bounded scalar/vector prediction.
    Architecture: word_embedding (obs -> scalar) + positional_embedding ->
    TransformerBlock -> flatten -> Linear -> Tanh * min_max_action.

    Used for supervised prediction: input is (N, seq_len, obs_size),
    output is (N, output_size) bounded by min_max_action.

    Attributes:
        word_embedding: Sequential(Linear -> ReLU -> Linear) reducing each obs to 1D.
        position_embedding: nn.Embedding(max_length, embed_size) for causality.
        layers: ModuleList with a single TransformerBlock.
        fc_out: Linear(embed_size * max_length, output_size) on flattened output.
        name: "Attention_Encoder" for checkpoint files.
    """
    def __init__(
            self,
            input_size: int,
            output_size: int,
            min_max_action: float,
            encode_size: int,
            embed_size: int,
            hidden_size: int,
            heads: int,
            forward_expansion: float,
            dropout: float,
            max_length: int,
    ) -> None:
        """Initialize AttentionEncoder.

        Args:
            input_size: Per-timestep observation dimensionality.
            output_size: Prediction dimensionality.
            min_max_action: Tanh output scaling factor.
            encode_size: Unused (kept for interface compatibility).
            embed_size: Embedding/attention dimension.
            hidden_size: Word embedding hidden layer size.
            heads: Number of attention heads.
            forward_expansion: Transformer feedforward multiplier.
            dropout: Dropout probability.
            max_length: Maximum sequence length (for positional embedding).
        """
        super().__init__()
        # masked_length is the max length of a sequence, so for us it is however long we want our sequences to be when training 
        self.min_max_action = min_max_action
        self.embed_size = embed_size
        self.device = get_device()
        self.word_embedding = nn.Sequential(nn.Linear(input_size, hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, 1))
        self.position_embedding = nn.Embedding(max_length, embed_size) # We need this to propagate the causality
        self.fc_out = nn.Linear(embed_size*max_length, output_size) # Transform to angle
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion,)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr = 0.0001  , weight_decay = 1e-4)
        self.to(self.device)

        self.name = 'Attention_Encoder'

    def forward(self, x: T.Tensor, mask: bool = None) -> T.Tensor:
        """Encode observation sequence into a bounded prediction.

        Args:
            x: Observation sequence of shape (N, seq_len, obs_size).
            mask: Optional attention mask.

        Returns:
            Prediction tensor of shape (N, output_size), bounded by min_max_action.
        """
        N, seq_len, obs_size = x.shape
        positions = T.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.word_embedding(x) + self.position_embedding(positions) 
        
        for layer in self.layers:
            mp = layer(out, out, out, mask)
        out = self.fc_out(mp.view(N,-1))
        # converts to single number in range (-mma, mma)
        out = self.tanh(out) * self.min_max_action
        return out
    
    def save_checkpoint(self, path: str) -> None:
        """ Save Model """
        print('... saving', self.name,'...')
        T.save(self.state_dict(), path + '_' + self.name)

    def load_checkpoint(self, path: str) -> None:
        """ Load Model """
        print('... loading', self.name, '...')
        if str(self.device) == 'cpu':
            self.load_state_dict(T.load(path + '_' + self.name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + self.name))