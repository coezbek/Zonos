# Based on gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/095b2229ee3a40e379c11f05b94bd6923db63b4b/model.py
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from zonos.config import BackboneConfig, InferenceParams

GLOBAL_ATTN = []
GLOBAL_AVERAGE = []

def precompute_freqs_cis(seq_len: int, n_elem: int, base: float = 10000) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def _update_kv_cache(
    k: torch.Tensor, v: torch.Tensor, inference_params: InferenceParams, layer_idx: int
) -> torch.Tensor:
    """k/v: (batch_size, seqlen, nheads, head_dim) or (batch_size, 1, nheads, head_dim)"""
    assert layer_idx in inference_params.key_value_memory_dict
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + k.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + k.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, 0, ...] = k
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, 1, ...] = v
    return kv_cache[batch_start:batch_end, :sequence_end, ...]

def highlight_phonemes(phoneme_string: str, rel_attention: torch.Tensor, threshold: float = 2.0) -> str:
    """
    phoneme_string: e.g. "hˈaloː lˈɔøtə!"
    rel_attention: e.g. shape [len(phoneme_string)], containing per-phoneme relative attention
    threshold: highlight if rel_attention[i] >= threshold

    returns: a string with each phoneme separated by space and 
             an asterisk marking the ones >= threshold
    """
    # We'll split the string into individual phonemes/chars. 
    # If your code uses a separate array of phonemes, adapt accordingly.
    # For raw text, this is just a list of each character.
    phonemes = list(phoneme_string)

    # Build an output list of tokens, highlighting as needed
    output_tokens = []
    for i, ch in enumerate(phonemes):
        # Safely handle if rel_attention is shorter than the string
        if i < len(rel_attention):
            if rel_attention[i] >= threshold:
                output_tokens.append(f"*{ch}*")
            else:
                output_tokens.append(f" {ch} ")
        else:
            # If we run out of attention values, just append the char
            output_tokens.append(ch)

    # Join with spaces (or however you want to separate them)
    return "".join(output_tokens)

class TorchZonosBackbone(nn.Module):
    supported_architectures = ["transformer"]
    freqs_cis: torch.Tensor

    def __init__(self, config: BackboneConfig):
        assert not config.ssm_cfg, "This backbone implementation only supports the Transformer model."
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config, i) for i in range(config.n_layer))
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        # TODO: This function should be pure
        head_dim = self.config.d_model // self.config.attn_cfg["num_heads"]
        self.freqs_cis = precompute_freqs_cis(16384, head_dim)
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states: torch.Tensor, inference_params: InferenceParams) -> torch.Tensor:
        input_pos = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
        # print(f"Before - Input pos shape: {input_pos.shape} - {input_pos}")
        input_pos = input_pos + inference_params.lengths_per_sample.unsqueeze(-1)
        # print(f"After - Input pos shape: {input_pos.shape} - {input_pos}")
        #print(f"Input pos: {input_pos[0, 0]}")
        
        freqs_cis = self.freqs_cis[input_pos].expand(hidden_states.shape[0], -1, -1, -1)

        GLOBAL_ATTN.clear()
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, inference_params, freqs_cis)

        if len(GLOBAL_ATTN) > 0:

            avg_attn_weights = torch.stack(GLOBAL_ATTN[2:3], dim=0).mean(0) # torch.stack(GLOBAL_ATTN, dim=0).mean(0)
            avg_attn_weights = avg_attn_weights / avg_attn_weights.sum(dim=-1, keepdim=True)

            if not hasattr(self, "running_attn_sum"):
                self.running_attn_sum = None
                self.running_count = 0

            if self.running_attn_sum is None:
                # First item
                self.running_attn_sum = avg_attn_weights.detach().clone()
                self.running_count = 1
            else:
                # Weighted running sum
                self.running_attn_sum += avg_attn_weights.detach()
                self.running_count += 1

            running_avg_attn = self.running_attn_sum / self.running_count

            avg_attn_weights = avg_attn_weights / running_avg_attn

            #top_phoneme_indices = avg_attn_weights.argmax(dim=-1)
            #for b_idx in range(0, avg_attn_weights.shape[0], 2):
            #    cond_idx = top_phoneme_indices[b_idx].item()
            ##    uncond_idx = top_phoneme_indices[b_idx + 1].item()
            #    print(f"Batch: {b_idx:2d}, Cond: {cond_idx:3d}, Uncond: {uncond_idx:3d}")

            values, indices = avg_attn_weights.topk(k=3, dim=-1)  # Each row: top 5 phoneme scores

            # print(f"Input pos: {input_pos[0, 0]:3d} - {avg_attn_weights[0, 0]:.3f} - {avg_attn_weights[0, 7]:.3f} {avg_attn_weights[0, 8]:.3f} {avg_attn_weights[0, 9]:.3f} - {avg_attn_weights[0, 16]:.3f} - {avg_attn_weights[0, 33]:.3f} - {avg_attn_weights[0, 39]:.3f} - {avg_attn_weights[0, 47]:.3f} - {indices[0].cpu().tolist()}")
            # HACK
            print(highlight_phonemes('hˈaloː lˈɔøtə! mˈaɪn nˈɑːmə ɪst kɾˈɪs svˈeːnaɪ', avg_attn_weights[0], threshold=2.0))


            # Now iterate over each batch example
#            for b_idx in range(0, avg_attn_weights.shape[0], 2):  # e.g., stepping in pairs
#                print(f"Input pos: {input_pos[0, 0]} - {b_idx} -> top_k {indices[b_idx]}")
                ## indices[b_idx] is shape [5], values[b_idx] is shape [5]
                #for rank in range(5):
                #    phoneme_idx = indices[b_idx, rank].item()
                #    attn_val = values[b_idx, rank].item()
                #    print(f"   Rank {rank+1}: phoneme idx {phoneme_idx} with attention {attn_val:.3f}")

        return self.norm_f(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config

        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mixer = Attention(config, layer_idx)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.mlp = FeedForward(config)

        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // config.attn_cfg["num_heads"]

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16):
        return torch.empty(batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim, dtype=dtype), None

    def forward(self, x: torch.Tensor, inference_params: InferenceParams, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm(x), inference_params, freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


class Attention(nn.Module):
    def __init__(self, config: BackboneConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.attn_cfg["num_heads"]
        self.num_heads_kv = config.attn_cfg["num_heads_kv"]
        self.head_dim = config.d_model // self.num_heads
        self.layer_idx = layer_idx

        total_head_dim = (self.num_heads + 2 * self.num_heads_kv) * self.head_dim
        self.in_proj = nn.Linear(config.d_model, total_head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, inference_params: InferenceParams, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seqlen, d_model = x.shape
        # print(f"B={batch_size}, T={seqlen}, D={d_model}")

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_heads_kv * self.head_dim
        q, k, v = self.in_proj(x).split([q_size, kv_size, kv_size], dim=-1)

        q = q.view(batch_size, seqlen, self.num_heads, self.head_dim)
        k = k.view(batch_size, seqlen, self.num_heads_kv, self.head_dim)
        v = v.view(batch_size, seqlen, self.num_heads_kv, self.head_dim)

        # [OPTIONAL] Debug prints or asserts
        # Assert Q and K are shaped as we expect:
        assert q.shape == (batch_size, seqlen, self.num_heads, self.head_dim), \
            f"Expected q to be [B={batch_size}, T={seqlen}, Hq={self.num_heads}, D={self.head_dim}], got {q.shape}"
        assert k.shape == (batch_size, seqlen, self.num_heads_kv, self.head_dim), \
            f"Expected k to be [B={batch_size}, T={seqlen}, Hk={self.num_heads_kv}, D={self.head_dim}], got {k.shape}"

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        kv = _update_kv_cache(k, v, inference_params, self.layer_idx)
        k, v = kv.unbind(dim=-3)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if False:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=seqlen > 1, enable_gqa=True)

        else:
            if self.num_heads_kv != self.num_heads:
                repeat_factor = self.num_heads // self.num_heads_kv
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)

            d_k = q.size(-1)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

            if seqlen > 1:
                causal_mask = torch.tril(torch.ones(seqlen, seqlen, device=q.device)).unsqueeze(0).unsqueeze(0)
                attn_logits = attn_logits.masked_fill(causal_mask == 0, float('-inf'))

            attn_weights = F.softmax(attn_logits, dim=-1)

            if attn_weights.shape[2] == 1:
                # Average over heads explicitly:
                avg_attn_weights = attn_weights.mean(dim=1).squeeze(1)  # [batch_size, src_len]
                avg_attn_weights = avg_attn_weights[:, :48] # Only consider the first 48 phonemes / HACK

                GLOBAL_ATTN.append(avg_attn_weights)

                # Now find the phoneme index with highest attention explicitly:
                #top_phoneme_indices = avg_attn_weights.argmax(dim=-1)  # [batch_size]

                # Explicitly print or store this information clearly:
                #for b_idx, phoneme_idx in enumerate(top_phoneme_indices):
                #    print(f"Batch {b_idx} is currently attending most strongly to phoneme index: {phoneme_idx.item()}")

                #if True or self.layer_idx == 25 or self.layer_idx == 0:
                #    for b_idx in range(0, avg_attn_weights.shape[0], 2):
                #        cond_idx = top_phoneme_indices[b_idx].item()
                #        uncond_idx = top_phoneme_indices[b_idx + 1].item()
                #        print(f"Layer: {self.layer_idx:2d}, Batch: {b_idx:2d}, Cond: {cond_idx:3d}, Uncond: {uncond_idx:3d}")
            else:
                print(f"Attention weights shape is not 2D (but: {attn_weights.shape}), skipping")

            y = torch.matmul(attn_weights, v)

        y = y.transpose(1, 2).contiguous().view(batch_size, seqlen, q_size)

        y = self.out_proj(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 2 * config.attn_mlp_d_intermediate, bias=False)
        self.fc2 = nn.Linear(config.attn_mlp_d_intermediate, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))
