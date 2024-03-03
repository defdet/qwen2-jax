import jax

from ..llama import Llama, LlamaModel
from ..llama.attention import Attention
from ..llama.decoder import Decoder
from .shard_array import shard_array

sharding_mp = Llama(
    model=LlamaModel(
        embedding=...,
        decoder=Decoder(
            input_norm=...,
            attention=Attention(q_proj=(1, 3), k_proj=(1, 2), v_proj=(1, 2), out_proj=(2, 4)),
            post_attn_norm=...,
            gate_proj=(1, 2),
            up_proj=(1, 2),
            down_proj=(2, 1),
        ),
        norm=...,
    ),
    lm_head=...,
)

def shard_model_params(params: Llama) -> Llama:
    return jax.tree_map(shard_array, params, sharding_mp)