import math
import logging

import torch
import torchaudio
from transformers.models.dac import DacModel

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    
    def load_prefix_audio(self, audio_path, device):
        """
        Loads and preprocesses the prefix audio (silence >= 100ms is recommended).
        Returns the encoded prefix codes.
        """
        wav, sr = torchaudio.load(audio_path)
        # Convert to mono by averaging channels if necessary.
        wav = wav.mean(dim=0, keepdim=True)
        wav = self.preprocess(wav, sr)
        wav = wav.to(device, dtype=torch.float32)
        
        # Add batch dimension before encoding.
        return self.encode(wav.unsqueeze(0))

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        assert codes.shape[1] == self.num_codebooks, f"Expected {self.num_codebooks} codebooks, got {codes.shape[1]}"
        with torch.autocast(self.dac.device.type, torch.float16, enabled=self.dac.device.type != "cpu"):
            return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1).float()

    def trim_silence(
        self,
        wav: torch.Tensor,
        threshold: float = 1e-5,
        frame_size: int = 512,
    ) -> torch.Tensor:
        """Trims leading and tailing silence using energy thresholding.
        
        Args:
            wav (torch.Tensor): Audio waveform (1D, mono, float32 in [-1, 1])
            threshold (float): Energy threshold for detecting speech start
            frame_size (int): Number of samples per frame to check energy over
        
        Returns:
            torch.Tensor: Trimmed waveform with leading and tailing silence removed
        """
        assert wav.ndim == 2 and wav.shape[0] == 1, "Expected mono audio tensor"
        
        num_frames = min((wav.shape[1] // frame_size) // 4, 16) # Limit to 16 frames ~ 180ms or max 1/4 of the audio
        
        # Check for leading silence
        start = 0
        for i in range(num_frames):
            frame = wav[:, i * frame_size : (i + 1) * frame_size]
            energy = frame.pow(2).mean()
            if energy > threshold:
                start = i * frame_size
                break

        # Check for tailing silence
        end = wav.shape[1]
        for i in range(num_frames):
            frame = wav[:, -((i + 1) * frame_size) : -i * frame_size]
            energy = frame.pow(2).mean()
            if energy > threshold:
                end = -((i + 1) * frame_size)
                break
        
        if start > 0 or end < wav.shape[1]:
            return wav[:, start:end]
        
        return wav
    
    _predictor = None
    
    def quality_string(self, input_aesthetics: dict[str, float]) -> str:
        """
        Converts a list of aesthetic scores into a formatted string with averages.
        Example output: "CE=4.2 CU=5.9 PC=2.1 PQ=6.8 AQ=4.8"
        """
        
        return " ".join([f"{i}={input_aesthetics[i]:.1f}" for i in input_aesthetics.keys()])

    def audio_quality(self, wavs: torch.Tensor | list[torch.Tensor], sr, qualities=['CU', 'CE', 'PQ', 'AQ']) -> dict[str, float]:
        """
        Compute the given Facebook Audio Aestethics of the given wav inputs

        Quality can be a list of:
            - 'PC' Production Quality 
            - 'CU' Content Usefulness
            - 'CE' Content Enjoyment
            - 'PC: Production Complexity - Disabled by default, because the generated audio isn't not complex
            - 'AQ' Average Quality of all selected metrics
        """
        
        if len(qualities) == 0:
            raise NotImplementedError("qualities must contain at least one of 'CU', 'CE', 'PQ', 'AQ', 'PC'")
        
        if not isinstance(wavs, list):
            wavs = [wavs]
        
        if self._predictor is None:
            from audiobox_aesthetics.infer import initialize_predictor
            self._predictor = initialize_predictor()

        # Compute aestethics from all audio files
        aesthetics = self._predictor.forward([{"path": wav, "sample_rate": sr} for wav in wavs])
        
        if qualities == ['AQ']:
            qualities_to_compute = ['CU', 'CE', 'PQ']
        else:
            qualities_to_compute = set(qualities) - {'AQ'}


        average_qualities = {j: sum(i[j] for i in aesthetics) / len(aesthetics) for j in qualities_to_compute}

        if 'AQ' in qualities:
            average_qualities['AQ'] = sum(average_qualities[j] for j in qualities_to_compute) / len(qualities_to_compute)

        return average_qualities
    
    def best_of_n(self, wavs: list[torch.Tensor], sr, n: int = -1) -> list[torch.Tensor]:
        """
        Select the best audio from a sample of size n from a list of audio tensors based on the average quality score.
        """
        
        n = len(wavs) if n == -1 else n

        return [max([wavs[i * n + j] for j in range(n)], key=lambda x: self.audio_quality(x, sr, qualities=['AQ'])['AQ']) for i in range(len(wavs) // n)]

    # Aim for -19.0 LUFS (Mono) compares to -16.0 LUFS (Stereo)
    def normalize_loudness(self, audio, sr, target_lufs=-19.0):
        
        try:
            import pyloudnorm
            # Set blocksize from 0.400 to 0.100 seconds for short generations
            block_size = 0.400 if audio.shape[1] > 2.0 * sr else 0.100
            loudness = pyloudnorm.Meter(sr, block_size=block_size).integrated_loudness(audio.cpu().numpy().T)
            gain_lufs = target_lufs - loudness
            gain = 10 ** (gain_lufs / 20.0)

            logger.debug(f"Adding gain to normalize loudness: {gain:.2f} dB")
            return audio * gain
        except Exception as e:
            logger.warning(f"Error normalizing loudness (audio too short?): {e}")
            return audio
        
    def codes_to_wavs(self, codes: torch.Tensor | list[torch.Tensor]) -> None:
        """
        Decode audio codes and into WAV file(s).
        
        Args:
            codes: Either a batched tensor [batch_size, num_codebooks, num_codes]
                  or a list of tensors each with shape [1, num_codebooks, num_codes] or [num_codebooks, num_codes]
        """
       
        # Handle different input types for codes and ensure we have a list of tensors of shape [1, num_codebooks, num_codes]
        if isinstance(codes, list):
            # Ensure each code has batch dimension
            code_list = []
            for c in codes:
                if c.dim() == 2:  # [num_codebooks, num_codes]
                    code_list.append(c.unsqueeze(0))
                else:  # [1, num_codebooks, num_codes] or [batch, num_codebooks, num_codes]
                    code_list.append(c)
        else:          
            if codes.dim() == 2:
                # [num_codebooks, num_codes] -> [1, num_codebooks, num_codes]
                code_list = [codes.unsqueeze(0)]
            elif codes.dim() == 3:
                # Batched tensor - split into list of individual tensors
                code_list = [codes[i:i+1] for i in range(codes.shape[0])]
            else:
                raise ValueError(f"Invalid shape for codes: {codes.shape}. Expected [num_codebooks, num_codes] or [batch_size, num_codebooks, num_codes]")
        
        results = []

        # Decode and save each audio file
        for c in code_list:
            # Decode codes to audio
            wav = self.decode(c).cpu().squeeze(0) # [batch==1, 1, num_samples] -> [1, num_samples]

            wav = self.normalize_loudness(wav, self.sampling_rate, -23.0)

            # Trim leading/tailing silence using simple energy stats
            wav = self.trim_silence(wav)

            # Add single block fade-in and fade-out over the first/last n samples to avoid clicks
            blocksize = 512
            num_blocks = 20 # 20/86 = 0.23 seconds
            wav[:,  :blocksize] *= torch.linspace(0, 1, blocksize, device=wav.device).unsqueeze(0) # fade-in

             # fade-out logarithmically from 10**0 to 10**-10 over num_blocks * blocksize samples
            num_blocks = min((wav.shape[1] // blocksize) // 4, 20) # Clamp to 20/86 = 0.23 seconds or 1/4 of the audio at most
            if num_blocks > 0:
                wav[:, -(num_blocks * blocksize):] *= torch.logspace(0, -10, num_blocks * blocksize, device=wav.device).unsqueeze(0)

            results.append(wav)
        
        return results
        
    def save_codes(self, paths: str | list[str], codes: torch.Tensor | list[torch.Tensor]) -> None:
        """
        Decode audio codes and save as WAV file(s).
        
        Args:
            path: A single output path or a list of output paths
            codes: Either a batched tensor [batch_size, num_codebooks, num_codes]
                  or a list of tensors each with shape [1, num_codebooks, num_codes] or [num_codebooks, num_codes]
        """

        if isinstance(paths, str):
            paths = [paths]

        wavs = self.codes_to_wavs(codes)

        # Ensure we have the right number of paths
        assert len(paths) == len(wavs), f"Number of paths ({len(paths)}) must match number of codes ({len(wavs)})"

        for p, w in zip(paths, wavs):
            # Save audio as WAV file
            torchaudio.save(p, w, self.sampling_rate)
            logger.debug(f"Saved audio to {p}")

