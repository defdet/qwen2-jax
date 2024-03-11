from functools import partial
import math
from typing import Any, NamedTuple

import einops as op
import jax
from jax import Array
import jax.nn as nn
import jax.numpy as jnp
import jax.random as rand

from .ModelConfig import ModelConfig
from .kv_cache import KVCache
from .rotary_embedding import RotaryValues, forward_rotary_embedding
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.pallas.ops.tpu import flash_attention
from jax.experimental.shard_map import shard_map
from .ring_attention import ring_attention
from .flash_attention import flash_attention, BlockSizes


class AttentionProj(NamedTuple):
    weight: Array
    bias: Array

class Attention(NamedTuple):
    q_proj: AttentionProj  # Array
    k_proj: AttentionProj  # Array
    v_proj: AttentionProj  # Array
    out_proj: AttentionProj  # Array

def check_attention(params: Attention, *, model_config: ModelConfig) -> None:
    assert isinstance(params.q_proj, Array)
    assert isinstance(params.k_proj, Array)
    assert isinstance(params.v_proj, Array)
    assert isinstance(params.out_proj, Array)

    assert params.q_proj.shape == (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k)
    assert params.k_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_k)
    assert params.v_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_v)
    assert params.out_proj.shape == (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model)

def init_attention(*, key: Array, model_config: ModelConfig) -> Attention:
    upper = 1. / math.sqrt(model_config.d_model)
    key0, key1, key2, key3 = rand.split(key, num=4)
    q_proj = rand.truncated_normal(key0, -upper, upper, (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k))
    k_proj = rand.truncated_normal(key1, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_k))
    v_proj = rand.truncated_normal(key2, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_v))
    out_proj = rand.truncated_normal(key3, -upper, upper, (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model))
    return Attention(q_proj, k_proj, v_proj, out_proj)

# Taken from EasyDel
def repeat_kv_bnsh(x: Array, n_rep: int) -> Array:
    """
    The repeat_kv_bnsh function is used to repeat the key and value vectors for each head in a multi-head attention
    module. This function takes as input an array of shape (batch_size, n_heads, sequence_length, head_dim) and returns
    an array of shape (batch_size, n_heads * nrep, sequence length, head dim). The reason this is necessary is because the
    attention module expects keys/values/queries to be repeated across heads but not across batches. However we want our
    keys/values/queries to be repeated both across heads AND batches so that we can use them

    :param x: chex.Array: Pass in the input to the function
    :param n_rep: int: Repeat the key and value heads
    :return: A new array with the same shape as x, except for the second dimension which is n_kv_heads * n_rep

    """
    bs, n_kv_heads, s, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, jax.numpy.newaxis, :, :]
    x = jax.numpy.repeat(x, n_rep, axis=2)

    return x.reshape(bs, n_kv_heads * n_rep, s, head_dim)

@partial(jax.jit, static_argnames=('model_config',))
def forward_attention(params: Attention, src_seq: Array, dst_seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    size_num = 128
    block_sizes = BlockSizes(
        block_q=size_num,
        block_k_major=size_num,
        block_k=size_num,
        block_b=1,
        block_q_major_dkv=size_num,
        block_k_major_dkv=size_num,
        block_k_dkv=size_num,
        block_q_dkv=size_num,
        block_k_major_dq=size_num,
        block_k_dq=size_num,
        block_q_dq=size_num,
    )
    attn_impl = 'normal'
    n_devices = jax.device_count()
    devices = mesh_utils.create_device_mesh((n_devices, ))
    if n_devices == 32:
        device_tuple = (4, 8)
    else:
        device_tuple = (2, n_devices // 2)

    q_axes = (0, 2)
    k_axes = (0, 1)
    v_axes = (0, 1)
    out_axes = (0, 2)

    sharding_tuple_q = [1] * 5
    sharding_tuple_k = [1] * 4
    sharding_tuple_v = [1] * 4
    sharding_tuple_out = [1] * 3

    for axis_num, axis in enumerate(q_axes):
        sharding_tuple_q[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(k_axes):
        sharding_tuple_k[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(v_axes):
        sharding_tuple_v[axis]=device_tuple[axis_num]
    for axis_num, axis in enumerate(out_axes):
        sharding_tuple_out[axis]=device_tuple[axis_num]

    sharding_tuple_q = tuple(sharding_tuple_q)
    sharding_tuple_k = tuple(sharding_tuple_k)
    sharding_tuple_v = tuple(sharding_tuple_v)
    sharding_tuple_out = tuple(sharding_tuple_out)
    
    name_tuple_q = tuple('abcdefghijklmnopqrstuvwxyz'[:5])
    mesh_q = Mesh(devices.reshape(sharding_tuple_q), name_tuple_q)     
    sharding_q = NamedSharding(mesh_q, P(*name_tuple_q))

    name_tuple_k = tuple('abcdefghijklmnopqrstuvwxyz'[:4])
    mesh_k = Mesh(devices.reshape(sharding_tuple_k), name_tuple_k)     
    sharding_k = NamedSharding(mesh_k, P(*name_tuple_k))

    name_tuple_v = tuple('abcdefghijklmnopqrstuvwxyz'[:4])
    mesh_v = Mesh(devices.reshape(sharding_tuple_v), name_tuple_v)     
    sharding_v = NamedSharding(mesh_v, P(*name_tuple_v))

    name_tuple_out = tuple('abcdefghijklmnopqrstuvwxyz'[:3])
    mesh_out = Mesh(devices.reshape(sharding_tuple_out), name_tuple_out)     
    sharding_out = NamedSharding(mesh_out, P(*name_tuple_out))

    q = op.einsum(src_seq, params.q_proj.weight, 'B S M, M R H K -> B R H S K')
    q += params.q_proj.bias.reshape(1, 1, q.shape[2], 1, q.shape[4])
    
    k = op.einsum(dst_seq, params.k_proj.weight, 'B D M, M H K -> B H D K')
    k += params.k_proj.bias.reshape(1, k.shape[1], 1, k.shape[3])
    
    v = op.einsum(dst_seq, params.v_proj.weight, 'B D M, M H V -> B H D V')
    v += params.v_proj.bias.reshape(1, v.shape[1], 1, k.shape[3])

    q = jax.lax.with_sharding_constraint(q, sharding_q)
    k = jax.lax.with_sharding_constraint(k, sharding_k)
    v = jax.lax.with_sharding_constraint(v, sharding_v)

    q = forward_rotary_embedding(q, rotary_values=rotary_values)
    k = forward_rotary_embedding(k, rotary_values=rotary_values)
    if kv_cache is not None:
            assert src_seq.shape[1] == 1
            assert dst_seq.shape[1] == 1
            k_cache, v_cache = kv_cache
            k = k_cache.at[:, :, -1:].set(k)
            v = v_cache.at[:, :, -1:].set(v)

    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)

    q_shape = q.shape
    if attn_impl == 'normal':
        qk = op.einsum(q, k, 'B R H S K, B H D K -> B R H S D')
        qk /= math.sqrt(model_config.d_k)
        qk = jnp.where(qk_mask, qk, -jnp.inf)
        qk = nn.softmax(qk)  # TODO: use `where`
        qk = jnp.where(qk_mask, qk, 0)  # TODO: why this line?

        qkv = op.einsum(qk, v, 'B R H S D, B H D V -> B R H S V')
        qkv = qkv.astype(jnp.bfloat16)
        out = op.einsum(qkv, params.out_proj.weight, 'B R H S V, R H V M -> B S M')
    else:

        # q = q.reshape(q.shape[0], model_config.n_rep_kv * model_config.n_heads_kv, q.shape[3], model_config.d_k)
        q = q.reshape(q_shape[0], q_shape[1] * q_shape[2], q_shape[3], q_shape[4]) # [B, H, S, K]
        q_shape = q.shape
    
        qk_mask = qk_mask.squeeze(1)
        qk_mask = jnp.broadcast_to(qk_mask, (qk_mask.shape[0], q_shape[1], q_shape[2], k.shape[2]))
    
    
        attention_bias = jax.lax.select(
                qk_mask == True,
                jnp.full(qk_mask.shape, 0.0).astype(jnp.bfloat16),
                jnp.full(qk_mask.shape, -10.0**6).astype(jnp.bfloat16),
            )
        specs_tuple = (P(*name_tuple_k),
                       P(*name_tuple_k),
                       P(*name_tuple_k),
                       P(*name_tuple_k))
        
        if attn_impl == 'flash':
            qkv = shard_map(partial(flash_attention, sm_scale=math.sqrt(model_config.d_k), debug=False, causal=False, block_sizes=block_sizes), mesh=mesh_k, in_specs=specs_tuple, out_specs=P(*name_tuple_k), check_rep=False)(q, k, v, attention_bias)
        if attn_impl == 'ring':
            qkv = shard_map(partial(ring_attention, sm_scale=math.sqrt(model_config.d_k), debug=False, causal=True), mesh=mesh_k, in_specs=specs_tuple, out_specs=P(*name_tuple_k), check_rep=False)(q, k, v, attention_bias)
            
        qkv = qkv.astype(jnp.bfloat16)
    
        qkv = qkv.reshape(qkv.shape[0], model_config.n_rep_kv, qkv.shape[1] // model_config.n_rep_kv, qkv.shape[2], -1)
        out = op.einsum(qkv, params.out_proj.weight, 'B R H S V, R H V M -> B S M') # Out proj has no bias
    out = jax.lax.with_sharding_constraint(out, sharding_out)
    
    kv_cache = None if not model_config.return_kv_cache else KVCache(k, v)

    return out, kv_cache
