import math
import logging

import torch
import torchaudio
from transformers.models.dac import DacModel

# Setup logger
logger = logging.getLogger(__name__)

class DACAutoencoder:
    def __init__(self):
        super().__init__()
        self.dac = DacModel.from_pretrained("descript/dac_44khz", local_files_only=False)
        self.dac.eval().requires_grad_(False)
        self.codebook_size = self.dac.config.codebook_size
        self.num_codebooks = self.dac.quantizer.n_codebooks
        self.sampling_rate = self.dac.config.sampling_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        """Resample and left-pad the input waveform for decoding with the DACAutoencoder."""
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        left_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
        return torch.nn.functional.pad(wav, (left_pad, 0), value=0)

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.dac.encode(wav).audio_codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.dac.device.type, torch.float16, enabled=self.dac.device.type != "cpu"):
            return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1).float()

    def save_codes(self, path: str | list[str], codes: torch.Tensor | list[torch.Tensor]) -> None:
        """
        Decode audio codes and save as WAV file(s).
        
        Args:
            path: A single output path or a list of output paths
            codes: Either a batched tensor [batch_size, num_codebooks, num_codes]
                  or a list of tensors each with shape [1, num_codebooks, num_codes] or [num_codebooks, num_codes]
        """
        # Convert single path to list for uniform handling
        paths = [path] if isinstance(path, str) else path
        
        # Handle different input types for codes
        if isinstance(codes, list):
            # Ensure each code has batch dimension
            code_list = []
            for c in codes:
                if c.dim() == 2:  # [num_codebooks, num_codes]
                    code_list.append(c.unsqueeze(0))
                else:  # [1, num_codebooks, num_codes] or [batch, num_codebooks, num_codes]
                    code_list.append(c)
        else:
            # Batched tensor - split into list of individual tensors
            code_list = [codes[i:i+1] for i in range(codes.shape[0])]
        
        # Ensure we have the right number of paths
        assert len(paths) == len(code_list), f"Number of paths ({len(paths)}) must match number of code tensors ({len(code_list)})"
        
        # Decode and save each audio file
        for p, c in zip(paths, code_list):
            # Decode codes to audio
            wav = self.decode(c).cpu()
            
            # Save audio as WAV file
            torchaudio.save(p, wav.squeeze(0), self.sampling_rate)
            logger.info(f"Saved audio to {p}")

