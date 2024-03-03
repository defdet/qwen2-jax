from operator import getitem
from typing import Callable

import jax
from jax import Array
import jax.nn as nn
import jax.numpy as jnp
import jax.random as rand

from ..rand_utils import split_key_nullable

# TODO: need type checking?
# _, seq_len = seq.shape
# assert seq.shape == (batch_size, seq_len)
# assert seq.dtype == jnp.uint16
# assert attn_mask.shape == (batch_size, seq_len)
# assert attn_mask.dtype == jnp.bool_

def PresencePenaltyProcessor(penalty: float) -> Callable:
    def inner(logits: Array, *, seq: Array, attn_mask: Array, **kwargs) -> Array:
        *_, vocab_size = logits.shape
        exists = jax.vmap(lambda row, row_mask: jnp.bincount(row, weights=row_mask.astype(jnp.uint16), length=vocab_size).astype(jnp.bool_))(seq, attn_mask)  # (batch_size, vocab_size)
        return logits - exists * penalty
    return inner

def FrequencyPenaltyProcessor(penalty: float) -> Callable:
    def inner(logits: Array, *, seq: Array, attn_mask: Array, **kwargs) -> Array:
        *_, vocab_size = logits.shape
        counts = jax.vmap(lambda row, row_mask: jnp.bincount(row, weights=row_mask.astype(jnp.uint16), length=vocab_size))(seq, attn_mask)  # (batch_size, vocab_size)
        return logits - counts * penalty
    return inner

def TopKSampler(top_k: int) -> Callable:
    def inner(logits: Array, *, key: Array, **kwargs) -> Array:
        batch_size, _ = logits.shape
        keys = rand.split(key, num=batch_size)

        def inner_inner(logits: Array, key: Array) -> Array:
            values, indices = jax.lax.top_k(logits, k=top_k)
            indices = indices.astype(jnp.uint16)
            selected_index = rand.categorical(key, values)
            selected_token_id = indices[selected_index]
            return selected_token_id

        selected_token_ids = jax.vmap(inner_inner)(logits, keys)
        return selected_token_ids
    setattr(inner, 'requires_key', True)
    return inner

def TopPSampler(top_p: float) -> Callable:
    def inner(logits: Array, *, key: Array, **kwargs) -> Array:
        batch_size, vocab_size = logits.shape

        indices = jnp.broadcast_to(jnp.arange(vocab_size, dtype=jnp.uint16), (batch_size, vocab_size))
        sorted_logits, sorted_indices = jax.lax.sort_key_val(-logits, indices, is_stable=False)
        sorted_logits = -sorted_logits
        sorted_probs = nn.softmax(sorted_logits)
        cum_probs = jnp.cumsum(sorted_probs, axis=-1)
        threshold_probs = jnp.maximum(cum_probs[:, 0], top_p)  # guarantee that at least one token will not be masked
        masked_sorted_logits = jnp.where(cum_probs >= threshold_probs[:, None], -jnp.inf, sorted_logits)

        key, subkey = rand.split(key)
        selected_indices = rand.categorical(subkey, masked_sorted_logits)
        selected_token_ids = jax.vmap(getitem)(sorted_indices, selected_indices)  # type: ignore[call-overload]
        return selected_token_ids
    setattr(inner, 'requires_key', True)
    return inner

def make_logits_processor(*callables):
    def inner(logits: Array, seq: Array, attn_mask: Array, key: Array | None) -> Array:
        for f in callables:
            if not getattr(f, 'requires_key', False):
                logits = f(logits, seq=seq, attn_mask=attn_mask)
            else:
                assert key is not None
                key, subkey = rand.split(key)
                logits = f(logits, seq=seq, attn_mask=attn_mask, key=subkey)
        return logits
    return inner
