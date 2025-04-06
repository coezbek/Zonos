import torch
import logging

# __name__ == "zonos.sampling"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def multinomial(input: torch.Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """

    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)

    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output

def summarize_tensor_stats(t: torch.Tensor, name="Tensor"):
    flat = t.flatten()
    stats = {
        "min": flat.min().item(),
        "q25": flat.quantile(0.25).item(),
        "median": flat.median().item(),
        "q75": flat.quantile(0.75).item(),
        "max": flat.max().item(),
        "mean": flat.mean().item(),
        "std": flat.std().item(),
    }

    headers = "  ".join(f"{k:>7}" for k in stats.keys())
    values  = "  ".join(f"{v:7.4f}" for v in stats.values())

    head = f"Stats for {name}"
    print(f"{head}: {headers}")
    print(f"{' ' * len(head)}  {values}")

def apply_unified(probs: torch.Tensor, linear: float, conf: float, quad: float, debug=False) -> torch.Tensor:
    """Sample next token using unified sampling approach that combines linear scaling, confidence, and quadratic terms.
    
    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        linear (float): Linear scaling factor applied to log probabilities.
        conf (float): Confidence factor that scales the entropy term.
        quad (float): Quadratic penalty factor applied to squared log probabilities.
    Returns:
        torch.Tensor: Modified probability distribution after applying unified sampling.
    """
    logprobs = torch.log(probs.clamp_min(1e-20))
    entropy = -torch.sum(probs * logprobs, dim=-1, keepdim=True)

    if debug:
        scaling = linear + entropy * conf - logprobs * quad
        summarize_tensor_stats(scaling[0, 0], f"unified scaling with {linear} linear and {entropy[0, 0].item():.4f} entropy")
        raw = logprobs * scaling
    else:
        raw = logprobs * (linear + entropy * conf) - logprobs**2 * quad

    return raw.softmax(dim=-1)

def apply_top_k(
    probs: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    v, _ = torch.topk(probs, min(k, probs.size(-1)))
    pivot = v.select(-1, -1).unsqueeze(-1)
    probs = torch.where(probs < pivot, 0.0, probs)
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def apply_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs = probs.scatter(-1, probs_idx, probs_sort)
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def apply_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Sample next token using min-p sampling.

    Args:
        scores (torch.FloatTensor): Input logits with token candidates on the last dimension.
        min_p (float): Minimum token probability, scaled by the probability of the most likely token.
                       Must be between 0 and 1. Typical values are in the 0.01-0.2 range.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    tokens_to_remove = probs < (min_p * top_probs)
    probs = probs.masked_fill(tokens_to_remove, 0.0)
    probs.div_(probs.sum(dim=-1, keepdim=True))
    return probs


def modify_logit_for_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    repetition_penalty: float,
    repetition_penalty_window: int,
):
    """See https://arxiv.org/abs/1909.05858
    Apply repetition penalty over a sliding window of the last `repetition_penalty_window` tokens.
    logits: (batch_size, n_codebooks, vocab_size)
    generated_tokens: (batch_size, n_codebooks, seq_len)
    """
    generated_tokens = generated_tokens[..., -repetition_penalty_window:]
    generated_tokens = generated_tokens.clamp_max(logits.shape[-1] - 1).to(torch.int64)
    rp = torch.full_like(logits, repetition_penalty)
    factors = torch.ones_like(logits).scatter_reduce(2, generated_tokens, rp, reduce="prod")

    # Debugging: print the penalized tokens
    if False:
        # Get positions where factor is different from 1 (i.e., penalized tokens)
        penalized_mask = factors != 1

        # Extract penalized token indices and their corresponding factors
        penalized_tokens = torch.nonzero(penalized_mask, as_tuple=True)  # Indices of penalized tokens
        penalized_values = factors[penalized_mask]  # The actual penalty values

        # Print penalized tokens per batch and codebook
        for batch_idx in range(logits.shape[0]):
            for codebook_idx in range(logits.shape[1]):
                mask = penalized_mask[batch_idx, codebook_idx]  # Mask for this batch/codebook
                if mask.any():  # If there are penalized tokens
                    tokens = generated_tokens[batch_idx, codebook_idx].tolist()
                    penalties = factors[batch_idx, codebook_idx][mask].tolist()
                    print(f"Batch {batch_idx}, Codebook {codebook_idx} | Penalized Tokens: {tokens} | Factors: {penalties}")
                
    return torch.where(logits <= 0, logits * factors, logits / factors)

def print_prob_stats(probs: torch.Tensor, batch_idx: int = 0, codebook_idx: int = 0, top_k: int = 5, mass_threshold: float = 0.95, before=False):
    p = probs[batch_idx, codebook_idx]
    top_probs, top_indices = torch.topk(p, k=top_k)
    num_non_zero = (p > 0).sum().item()
    sorted_p, _ = torch.sort(p, descending=True)
    cumulative = torch.cumsum(sorted_p, dim=0)
    tokens_to_mass = (cumulative < mass_threshold).sum().item() + 1
    tokens_str = ', '.join(f'{t:>4}' for t in top_indices.tolist())
    probs_str = ', '.join(f'{v:.3f}' for v in top_probs.tolist())

    before_str = "Before" if before else "After "
    print(f"{before_str} Batch {batch_idx}, Codebook {codebook_idx} | Top {top_k}: [{tokens_str}] | Probs: [{probs_str}] | Non-zero: {num_non_zero:>4} | {int(mass_threshold*100)}% mass in: {tokens_to_mass:>4} tokens | {before_str}")
    if not before:
        global distribution
        distribution.append(tokens_to_mass)
        print(f"  Average number of tokens to choose top 95%: {sum(distribution) / len(distribution):.2f}")

        global num_non_zero_tokens
        num_non_zero_tokens.append(num_non_zero)
        print(f"  Average number of non-zero tokens: {sum(num_non_zero_tokens) / len(num_non_zero_tokens):.2f}")
    
offset = 0
distribution = []
num_non_zero_tokens = []

def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    linear: float = 0.0,
    conf: float = 0.0,
    quad: float = 0.0,
    generated_tokens: torch.Tensor | None = None,
    repetition_penalty: float = 3.0,
    repetition_penalty_window: int = 2,
) -> torch.Tensor:
    """Sample next token from logits using either top_k/p/min_p OR using NovelAI's Unified Sampler.
    
    Args:
        logits (torch.Tensor): Input logits with token candidates on the last dimension.

        temperature (float): Randomness of the sampling. Lower temperature results in more deterministic samples.
            To disable sampling entirely, set it to 0. For NovelAI's Unified Sampler, set it to 1.0

        top_p (float): Only sample from the most probable tokens whose cumulative probability is less than p.
            This is called nucleus sampling. Must be between 0 and 1. Typical values are in the 0.1-0.9 range.

            Set to 0 to disable.

        top_k (int): Only sample from the top k most probable tokens. Set to 0 to disable.
        
        min_p (float): Minimum token probability, scaled by the probability of the most likely token.
                       Must be between 0 and 1. Typical values are in the 0.01-0.2 range.
                       If too high, no token might be sampled leading to silence (?)

        linear (float): NovelAI's Unified Sampler -> 0.0 to 1.0, default from gradio 0.5

            Set Linear between 0 and 1 according to how unusual you want tokens to be. 
            Lower numbers will produce more unusual/creative outputs, 
            but you will have to reroll or edit more. 

        conf (float): Confidence - Low values make random outputs more random. -> -2.0 * Quad to 2.0, default from gradio 0.4

            As a starting point, set Quad = 1/3 - Linear * 4 / 15, and Conf = -Quad / 2.

        quad (float): Quadratic - High values make low probablities much lower. -> -2.0 to 2.0, default from gradio 0.0

    Returns:
        torch.Tensor: Sampled tokens.
    """
    if repetition_penalty != 1.0 and generated_tokens is not None:
        logits = modify_logit_for_repetition_penalty(logits, generated_tokens, repetition_penalty, repetition_penalty_window)

    global offset
    if offset == 0 and logger.isEnabledFor(logging.DEBUG):
        print(f"Temperature: {temperature}, Top P: {top_p}, Top K: {top_k}, Min P: {min_p}, Linear: {linear}, Conf: {conf}, Quad: {quad} | RepPen: {repetition_penalty}, RepPenWindow: {repetition_penalty_window}")

    if temperature > 0:

        # Linear disables temperature
        probs = torch.softmax(logits / temperature, dim=-1)
        offset += 1
        debug = offset % 64 == 0 and logger.isEnabledFor(logging.DEBUG)

        if top_p > 0:
            mass_threshold = top_p
        else:
            mass_threshold = 0.95

        if debug:
            print_prob_stats(probs, batch_idx=0, codebook_idx=0, top_k=5, mass_threshold=mass_threshold, before=True)

        if linear > 0:
            probs = apply_unified(probs, linear, conf, quad, debug=debug)
        
        if top_p > 0:
            probs = apply_top_p(probs, top_p)
        if top_k > 0:
            probs = apply_top_k(probs, top_k)
        if min_p > 0:
            probs = apply_min_p(probs, min_p)

        # Only print for codebook 0
        if debug:
            print_prob_stats(probs, batch_idx=0, codebook_idx=0, top_k=5, mass_threshold=mass_threshold, before=False)

        next_token = multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    return next_token  # [batch_size, num_codebooks, 1]
