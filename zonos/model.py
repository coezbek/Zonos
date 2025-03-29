import json
from typing import Callable
import logging
import safetensors
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from zonos.autoencoder import DACAutoencoder
from zonos.backbone import BACKBONES
from zonos.codebook_pattern import apply_delay_pattern, revert_delay_pattern
from zonos.conditioning import PrefixConditioner
from zonos.config import InferenceParams, ZonosConfig
from zonos.sampling import sample_from_logits
from zonos.speaker_cloning import SpeakerEmbeddingLDA
from zonos.utils import DEFAULT_DEVICE, find_multiple, pad_weight_

DEFAULT_BACKBONE_CLS = next(iter(BACKBONES.values()))


class Zonos(nn.Module):
    def __init__(self, config: ZonosConfig, backbone_cls=DEFAULT_BACKBONE_CLS):
        super().__init__()
        self.config = config
        dim = config.backbone.d_model
        self.eos_token_id = config.eos_token_id
        self.masked_token_id = config.masked_token_id

        self.autoencoder = DACAutoencoder()
        self.backbone = backbone_cls(config.backbone)
        self.prefix_conditioner = PrefixConditioner(config.prefix_conditioner, dim)
        self.spk_clone_model = None

        # TODO: pad to multiple of at least 8
        self.embeddings = nn.ModuleList([nn.Embedding(1026, dim) for _ in range(self.autoencoder.num_codebooks)])
        self.heads = nn.ModuleList([nn.Linear(dim, 1025, bias=False) for _ in range(self.autoencoder.num_codebooks)])

        self._cg_graph = None
        self._cg_batch_size = None
        self._cg_input_ids = None
        self._cg_logits = None
        self._cg_inference_params = None
        self._cg_scale = None

        if config.pad_vocab_to_multiple_of:
            self.register_load_state_dict_post_hook(self._pad_embeddings_and_heads)

    def _pad_embeddings_and_heads(self, *args, **kwargs):
        for w in [*self.embeddings, *self.heads]:
            pad_weight_(w, self.config.pad_vocab_to_multiple_of)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls, repo_id: str, revision: str | None = None, device: str = DEFAULT_DEVICE, **kwargs
    ) -> "Zonos":
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision, local_files_only=False)
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=revision, local_files_only=False)
        return cls.from_local(config_path, model_path, device, **kwargs)

    @classmethod
    def from_local(
        cls, config_path: str, model_path: str, device: str = DEFAULT_DEVICE, backbone: str | None = None
    ) -> "Zonos":
        config = ZonosConfig.from_dict(json.load(open(config_path)))
        if backbone:
            backbone_cls = BACKBONES[backbone]
        else:
            is_transformer = not bool(config.backbone.ssm_cfg)
            backbone_cls = DEFAULT_BACKBONE_CLS
            # Preferentially route to pure torch backbone for increased performance and lower latency.
            if is_transformer and "torch" in BACKBONES:
                backbone_cls = BACKBONES["torch"]

        model = cls(config, backbone_cls).to(device, torch.bfloat16)
        model.autoencoder.dac.to(device)

        sd = model.state_dict()
        with safetensors.safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        model.load_state_dict(sd)

        return model

    def make_speaker_embedding(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Generate a speaker embedding from an audio clip."""
        if self.spk_clone_model is None:
            self.spk_clone_model = SpeakerEmbeddingLDA()
        _, spk_embedding = self.spk_clone_model(wav.to(self.spk_clone_model.device), sr)
        return spk_embedding.unsqueeze(0).bfloat16()

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return sum(emb(codes[:, i]) for i, emb in enumerate(self.embeddings))

    def apply_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(hidden_states) for head in self.heads], dim=1)

    def _compute_logits(
        self, hidden_states: torch.Tensor, inference_params: InferenceParams, cfg_scale: float
    ) -> torch.Tensor:
        """
        Pass `hidden_states` into `backbone` and `multi_head`, applying
        classifier-free guidance if `cfg_scale != 1.0`.
        """
        last_hidden_states = self.backbone(hidden_states, inference_params)[:, -1, :].unsqueeze(1)
        logits = self.apply_heads(last_hidden_states).squeeze(2).float()
        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        logits[..., 1025:].fill_(-torch.inf)  # ensures padding is ignored
        return logits

    def _decode_one_token(
        self,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
        allow_cudagraphs: bool = True,
    ) -> torch.Tensor:
        """
        Single-step decode. Prepares the hidden states, possibly replicates them
        for CFG, and then delegates to `_compute_logits`.

        Below we wrap this function with a simple CUDA Graph capturing mechanism,
        doing 3 warmup steps if needed and then capturing or replaying the graph.
        We only recapture if the batch size changes.
        """
        # TODO: support cfg_scale==1
        if cfg_scale == 1.0:
            hidden_states = self.embed_codes(input_ids)
            return self._compute_logits(hidden_states, inference_params, cfg_scale)

        bsz = input_ids.size(0)

        if not allow_cudagraphs or input_ids.device.type != "cuda":
            hidden_states_local = self.embed_codes(input_ids)
            hidden_states_local = hidden_states_local.repeat(2, 1, 1)
            return self._compute_logits(hidden_states_local, inference_params, cfg_scale)

        need_capture = (self._cg_graph is None) or (self._cg_batch_size != bsz)

        if need_capture:
            self._cg_graph = None

            self._cg_batch_size = bsz
            self._cg_inference_params = inference_params
            self._cg_scale = cfg_scale

            for _ in range(3):
                hidden_states = self.embed_codes(input_ids)
                hidden_states = hidden_states.repeat(2, 1, 1)  # because cfg != 1.0
                logits = self._compute_logits(hidden_states, inference_params, cfg_scale)

            self._cg_input_ids = input_ids.clone()
            self._cg_logits = torch.empty_like(logits)

            g = torch.cuda.CUDAGraph()

            def capture_region():
                hidden_states_local = self.embed_codes(self._cg_input_ids)
                hidden_states_local = hidden_states_local.repeat(2, 1, 1)
                self._cg_logits = self._compute_logits(hidden_states_local, self._cg_inference_params, self._cg_scale)

            with torch.cuda.graph(g):
                capture_region()

            self._cg_graph = g

        else:
            self._cg_input_ids.copy_(input_ids)

        self._cg_graph.replay()

        return self._cg_logits

    def _prefill(
        self,
        prefix_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        inference_params: InferenceParams,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        "Prefill" mode: we already have `prefix_hidden_states`, and we want
        to append new embeddings, then compute the logits.
        """
        # Replicate input_ids if CFG is enabled
        #if cfg_scale != 1.0:
        #    input_ids = input_ids.expand(prefix_hidden_states.shape[0], -1, -1)

        if input_ids.shape[0] != prefix_hidden_states.shape[0]:
            # Calculate the required duplication factor.
            factor = prefix_hidden_states.shape[0] // input_ids.shape[0]
            input_ids = input_ids.repeat(factor, 1, 1)

        hidden_states = torch.cat([prefix_hidden_states, self.embed_codes(input_ids)], dim=1)
        return self._compute_logits(hidden_states, inference_params, cfg_scale)

    def setup_cache(self, batch_size: int, max_seqlen: int, dtype: torch.dtype = torch.bfloat16) -> InferenceParams:
        max_seqlen = find_multiple(max_seqlen, 8)
        key_value_memory_dict = self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        lengths_per_sample = torch.full((batch_size,), 0, dtype=torch.int32)
        return InferenceParams(max_seqlen, batch_size, 0, 0, key_value_memory_dict, lengths_per_sample)

    def prepare_conditioning(self, cond_dict: dict, uncond_dict: dict | None = None) -> torch.Tensor:
        if uncond_dict is None:
            uncond_dict = {k: cond_dict[k] for k in self.prefix_conditioner.required_keys}
        return torch.cat(
            [
                self.prefix_conditioner(cond_dict),
                self.prefix_conditioner(uncond_dict),
            ]
        )

    def can_use_cudagraphs(self) -> bool:
        # Only the mamba-ssm backbone supports CUDA Graphs at the moment
        return self.device.type == "cuda" and "_mamba_ssm" in str(self.backbone.__class__)

    @torch.inference_mode()
    def generate(
        self,
        prefix_conditioning: torch.Tensor,  # [batch_size, cond_seq_len, d_model]
        audio_prefix_codes: torch.Tensor | None = None,  # [batch_size, 9, prefix_audio_seq_len]
        max_new_tokens: int = 86 * 30,
        cfg_scale: float = 2.0,
        batch_size: int = 1,
        sampling_params: dict = dict(
            top_p=0, 
            top_k=0, 
            min_p=0,           
            linear=0.55, 
            conf=0.4, 
            quad=0.0, 
            repetition_penalty=3.0,
            repetition_penalty_window=2,
            temperature=1.0),
        progress_bar: bool = True,
        disable_torch_compile: bool = False,
        callback: Callable[[torch.Tensor, int, int], bool] | None = None,
    ):
        # Print the sampling parameters as a short string
        sampling_params_str = '_'.join([f"{k[0]}{v}" for k, v in sampling_params.items()])
        logging.debug(f"Sampling parameters: {sampling_params_str}")
        
        # Ensure cfg_scale is supported (avoid cfg_scale = 1)
        assert cfg_scale != 1, "TODO: add support for cfg_scale=1"

        # Determine length of any provided audio prefix
        # (audio_prefix_codes is [batch_size, 9, prefix_audio_seq_len])
        prefix_audio_len = 0 if audio_prefix_codes is None else audio_prefix_codes.shape[2]

        # Get the device (CPU or GPU) on which the model is running
        device = self.device

        # Check feasibility of CUDA Graphs and possibly torch.compile
        cg = self.can_use_cudagraphs()

        # Compile the _decode_one_token function
        decode_one_token = self._decode_one_token
        decode_one_token = torch.compile(decode_one_token, dynamic=True, disable=cg or disable_torch_compile)

        # Special token ID for 'unknown' (filler values in the codes array)
        unknown_token = -1

        # Compute how long the final audio sequence can be
        audio_seq_len = prefix_audio_len + max_new_tokens

        # The sequence length includes text prefix, the audio sequence, plus 9 additional ones
        seq_len = prefix_conditioning.shape[1] + audio_seq_len + 9

        # Set up the inference cache for the model (key-value caching, etc.)
        with torch.device(device):
            inference_params = self.setup_cache(batch_size=batch_size * 2, max_seqlen=seq_len)
            codes = torch.full((batch_size, 9, audio_seq_len), unknown_token)

        # If audio prefix codes exist, copy them into the codes tensor
        if audio_prefix_codes is not None:
            codes[..., :prefix_audio_len] = audio_prefix_codes

        # Apply a delay pattern to the codes (typically reordering or shifting tokens)
        delayed_codes = apply_delay_pattern(codes, self.masked_token_id)

        # Select the delayed prefix slice from the start through prefix_audio_len+1
        delayed_prefix_audio_codes = delayed_codes[..., : prefix_audio_len + 1]

        # Prefill the model with the text prefix and the initial delayed audio codes
        logits = self._prefill(prefix_conditioning, delayed_prefix_audio_codes, inference_params, cfg_scale)

        # Sample one token from the logits
        next_token = sample_from_logits(logits, **sampling_params)

        # The offset marks how many tokens we have processed in the audio dimension
        offset = delayed_prefix_audio_codes.shape[2]

        # Retrieve the next "frame" (the [batch_size, 9, 1] slice) and fill unknown positions
        frame = delayed_codes[..., offset : offset + 1]
        # For multiple batches, we can't use frame.masked_scatter_(frame == unknown_token, next_token) 
        # because it is continuing one-by-one for each unmasked entry
        # going across batches
        mask = (frame == unknown_token)
        frame.masked_scatter_(mask, next_token[mask])

        # Update inference parameters to account for the initial tokens
        prefix_length = prefix_conditioning.shape[1] + prefix_audio_len + 1
        inference_params.seqlen_offset += prefix_length
        inference_params.lengths_per_sample[:] += prefix_length

        # Create a logit bias that forbids codebooks 1..8 from generating an EOS token
        logit_bias = torch.zeros_like(logits)
        logit_bias[:, 1:, self.eos_token_id] = -torch.inf  # only allow codebook 0 to predict EOS
        logit_bias[:, 0, self.eos_token_id] -= torch.log(torch.tensor(2.0, device=logits.device)) # Make EOS less likely because audio often is cut off

        # Track which samples have stopped (hit EOS) and how many steps remain
        stopping = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_steps = delayed_codes.shape[2] - offset
        remaining_steps = torch.full((batch_size,), max_steps, device=device)
        progress = tqdm(total=max_steps, desc="Generating", disable=not progress_bar)

        cfg_scale = torch.tensor(cfg_scale)
        step = 0

        # Main loop to decode tokens until all sequences have reached their limit or hit EOS
        while torch.max(remaining_steps) > 0:
            offset += 1

            # Retrieve the latest input token
            input_ids = delayed_codes[..., offset - 1 : offset]

            # Decode one token's logits and add the logit bias
            logits = decode_one_token(input_ids, inference_params, cfg_scale, allow_cudagraphs=cg)
            logits += logit_bias

            # Sample from these logits
            next_token = sample_from_logits(
                logits,
                generated_tokens=delayed_codes[..., :offset],
                **sampling_params
            )

            # Check if any samples have produced an EOS in codebook 0
            eos_in_cb0 = next_token[:, 0] == self.eos_token_id

            # If EOS in codebook 0, limit the remaining steps for that sample
            remaining_steps[eos_in_cb0[:, 0]] = torch.minimum(
                remaining_steps[eos_in_cb0[:, 0]], torch.tensor(9)
            )
            stopping |= eos_in_cb0[:, 0]

            # Determine the codebook index in which EOS should be placed
            eos_codebook_idx = 9 - remaining_steps
            eos_codebook_idx = torch.clamp(eos_codebook_idx, max=9 -1)

            # For any samples that are stopping, fill all preceding codebooks with a masked token
            # and place an EOS token in the correct codebook
            for i in range(next_token.shape[0]):
                if stopping[i]:
                    idx = eos_codebook_idx[i].item()
                    next_token[i, :idx] = self.masked_token_id
                    next_token[i, idx] = self.eos_token_id

            # Insert the newly sampled tokens into the delayed_codes array
            frame = delayed_codes[..., offset : offset + 1]
            mask = (frame == unknown_token)
            frame.masked_scatter_(mask, next_token[mask])
            
            # Advance the model's KV cache tracking
            inference_params.seqlen_offset += 1
            inference_params.lengths_per_sample[:] += 1

            # Decrement the available steps for each sample
            remaining_steps -= 1

            # Update the progress bar and step counter
            progress.update()
            step += 1

            # If a callback is defined, call it; if it returns False, we break
            if callback is not None and not callback(frame, step, max_steps):
                break

        # Close the progress bar
        progress.close()

        # Revert the delay pattern to restore normal sequential ordering
        out_codes = revert_delay_pattern(delayed_codes)

        # Find the first EOS token for each sample in codebook 0
        eos_positions = (out_codes[:, 0, :] == self.eos_token_id).int().argmax(dim=-1)

        # Slice off anything beyond offset - 9
        out_codes = out_codes[..., : offset - 9]

        # Mask out invalid tokens (>= 1024) to 0
        out_codes.masked_fill_(out_codes >= 1024, 0)

        # Trim each sequence at its own EOS position and store in a list
        out_codes_list = [out_codes[i, :, :eos_positions[i]].clone() for i in range(out_codes.shape[0])]

        # Reset internal CUDA graph if used
        self._cg_graph = None

        # Return list of variable-length sequences
        return out_codes_list