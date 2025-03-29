import argparse
import torch
import torchaudio
import os
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

def load_audio(file_path, model):
    """Loads and preprocesses audio."""
    wav, sr = torchaudio.load(file_path)
    wav = wav.mean(0, keepdim=True)  # Convert to mono if needed
    wav = model.autoencoder.preprocess(wav, sr)
    return model.autoencoder.encode(wav.unsqueeze(0).to(device, dtype=torch.float32))

def generate_audio(args, model, speaker_embedding, prefix_audio_codes):
    """Generates speech audio using Zonos."""
    torch.manual_seed(args.seed)
    
    cond_dict = make_cond_dict(
        text=args.text,
        speaker=speaker_embedding,
        language=args.language,
        emotion=args.emotion,
        fmax=args.fmax,
        pitch_std=args.pitch_std,
        speaking_rate=args.speaking_rate,
        vqscore_8=args.vqscore_8,
        ctc_loss=args.ctc_loss,
        dnsmos_ovrl=args.dnsmos_ovrl,
        speaker_noised=args.speaker_noised,
        unconditional_keys=args.unconditional_keys,
    )
    
    conditioning = model.prepare_conditioning(cond_dict)
    
    codes = model.generate(
        conditioning,
        audio_prefix_codes=prefix_audio_codes,
        max_new_tokens=args.max_new_tokens,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        sampling_params={
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "linear": args.linear,
            "conf": args.conf,
            "quad": args.quad,
            "repetition_penalty": args.repetition_penalty,
            "repetition_penalty_window": args.repetition_penalty_window,
            "temperature": args.temperature,
        },
        progress_bar=args.progress_bar,
    )
    
    wavs = model.autoencoder.decode(codes).cpu()
    torchaudio.save(args.output, wavs[0], model.autoencoder.sampling_rate)
    print(f"Generated audio saved to {args.output}")

def main():
    parser = argparse.ArgumentParser(description="Generate speech with Zonos CLI.")
    parser.add_argument("--text", required=True, help="Text to generate speech for.")
    parser.add_argument("--language", required=True, help="Language code (e.g., en-us, de).")
    parser.add_argument("--reference_audio", default="assets/exampleaudio.mp3", help="Path to reference speaker audio (default: example audio).")
    parser.add_argument("--prefix_audio", default=None, help="Path to prefix audio (default: 100ms silence).")
    parser.add_argument("--output", default="output.wav", help="Output wav file name.")
    parser.add_argument("--seed", type=int, default=423, help="Random seed for reproducibility.")
    
    # Conditioning parameters
    parser.add_argument("--emotion", nargs=8, type=float, default=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2], help="Emotion vector (Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral).")
    parser.add_argument("--fmax", type=float, default=22050.0, help="Max frequency (0-24000).")
    parser.add_argument("--pitch_std", type=float, default=45.0, help="Pitch standard deviation (0-400).")
    parser.add_argument("--speaking_rate", type=float, default=15.0, help="Speaking rate (0-40).")
    parser.add_argument("--vqscore_8", nargs=8, type=float, default=[0.78] * 8, help="VQScore per 1/8th of audio (hybrid-only).")
    parser.add_argument("--ctc_loss", type=float, default=0.0, help="CTC loss target (hybrid-only).")
    parser.add_argument("--dnsmos_ovrl", type=float, default=4.0, help="DNSMOS overall score (hybrid-only).")
    parser.add_argument("--speaker_noised", action='store_true', help="Apply speaker noise (hybrid-only).")
    parser.add_argument("--unconditional_keys", nargs='*', default=["emotion", "vqscore_8", "dnsmos_ovrl"], help="Unconditional keys.")
    
    # Model generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=86 * 30, help="Max new tokens.")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="CFG scale.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--top_p", type=float, default=0, help="Top-p sampling.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling.")
    parser.add_argument("--min_p", type=float, default=0, help="Minimum probability threshold.")
    parser.add_argument("--linear", type=float, default=0.8, help="Linear scaling factor.")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence parameter.")
    parser.add_argument("--quad", type=float, default=0.0, help="Quadratic factor.")
    parser.add_argument("--repetition_penalty", type=float, default=1.5, help="Repetition penalty.")
    parser.add_argument("--repetition_penalty_window", type=int, default=8, help="Repetition penalty window.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling.")
    parser.add_argument("--progress_bar", default=True, action='store_true', help="Show progress bar.")
    parser.add_argument("--verbose", action='store_true', help="Enable debug printouts.")
    
    args = parser.parse_args()

    if args.verbose:
      import logging
      # To enable all DEBUG logging:
      logging.basicConfig(level=logging.DEBUG)
      # Or enable only phonemizer DEBUG logging:
      logging.getLogger("phonemizer").setLevel(logging.DEBUG)
    
    print("Loading Zonos model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    
    print("Loading speaker reference audio...")
    wav, sr = torchaudio.load(args.reference_audio)
    speaker_embedding = model.make_speaker_embedding(wav, sr)
    
    print("Loading prefix audio...")
    if args.prefix_audio:
        prefix_audio_codes = load_audio(args.prefix_audio, model)
    else:
        silence_path = "assets/silence_100ms.wav"  # Ensure this file exists
        prefix_audio_codes = load_audio(silence_path, model)
    
    print("Generating speech...")
    generate_audio(args, model, speaker_embedding, prefix_audio_codes)
    
if __name__ == "__main__":
    main()