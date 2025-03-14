import argparse
import torch
import torchaudio
import torch.nn.functional as F
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
    """Generates speech for multiple texts in a batch."""
    torch.manual_seed(args.seed)
    batch_size = len(args.text)
    
    # Expand speaker embedding for batch size
    # speaker_embedding = speaker_embedding.expand(batch_size, speaker_embedding.size(1), speaker_embedding.size(2))
    # speaker_embedding = speaker_embedding.expand(batch_size, -1)

    # Create conditioning dictionaries for each text
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

    prefix_conditioning = model.prepare_conditioning(cond_dict)
        
    # Expand prefix_audio_codes to batch size
    prefix_audio_codes = prefix_audio_codes.expand(batch_size, prefix_audio_codes.size(1), prefix_audio_codes.size(2))

    print ("conditioning.size(): ", prefix_conditioning.size())
    print ("speaker_embedding.size(): ", speaker_embedding.size())
    print ("prefix_audio_codes.size(): ", prefix_audio_codes.size())
    
    # Generate codes
    codes = model.generate(
        prefix_conditioning,
        audio_prefix_codes=prefix_audio_codes,
        max_new_tokens=args.max_new_tokens,
        cfg_scale=args.cfg_scale,
        batch_size=batch_size,
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

    # Decode and save batch results
    # wavs = model.autoencoder.decode(codes).cpu()

    for i, code in enumerate(codes):
        # Ensure code has the correct dimensions: [1, num_codebooks, sequence_length]
        code = code.unsqueeze(0) if code.dim() == 2 else code
        # Decode the code
        decoded_audio = model.autoencoder.decode(code).cpu()

        # Save the decoded audio
        output_file = f"{args.output.rstrip('.wav')}_{i}.wav"
        torchaudio.save(output_file, decoded_audio.squeeze(0), model.autoencoder.sampling_rate)
        print(f"Generated audio saved to {output_file}")

    #for i, wav in enumerate(wavs):
    #    output_file = f"{args.output.rstrip('.wav')}_{i}.wav"
    #    torchaudio.save(output_file, wav, model.autoencoder.sampling_rate)
    #    print(f"Generated audio saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate speech with Zonos CLI (Batch Mode).")
    parser.add_argument("--text", nargs="+", required=True, help="List of texts to generate speech for.")
    parser.add_argument("--language", required=True, help="Language code (e.g., en-us, de).")
    parser.add_argument("--reference_audio", default="assets/exampleaudio.mp3", help="Path to reference speaker audio.")
    parser.add_argument("--prefix_audio", default=None, help="Path to prefix audio (default: 100ms silence).")
    parser.add_argument("--output", default="output.wav", help="Output wav file prefix.")
    parser.add_argument("--seed", type=int, default=423, help="Random seed for reproducibility.")
 
    # Conditioning parameters
    parser.add_argument("--emotion", nargs=8, type=float, default=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2], help="Emotion vector (Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral).")
    parser.add_argument("--fmax", type=float, default=22050.0, help="Max frequency (0-24000).")
    parser.add_argument("--pitch_std", type=float, default=45.0, help="Pitch standard deviation (0-400).")
    parser.add_argument("--speaking_rate", type=float, default=15.0, help="Speaking rate (0-40).")
    parser.add_argument("--vqscore_8", nargs=8, type=float, default=[0.78] * 8, help="VQScore per 1/8th of audio.")
    parser.add_argument("--ctc_loss", type=float, default=0.0, help="CTC loss target.")
    parser.add_argument("--dnsmos_ovrl", type=float, default=4.0, help="DNSMOS overall score.")
    parser.add_argument("--speaker_noised", action='store_true', help="Apply speaker noise.")
    parser.add_argument("--unconditional_keys", nargs='*', default=["emotion", "vqscore_8", "dnsmos_ovrl"], help="Unconditional keys.")
        
    # Model generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=86 * 30, help="Max new tokens.")
    parser.add_argument("--cfg_scale", type=float, default=2.0, help="CFG scale.")
    parser.add_argument("--top_p", type=float, default=0, help="Top-p sampling.")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling.")
    parser.add_argument("--min_p", type=float, default=0, help="Minimum probability threshold.")
    parser.add_argument("--linear", type=float, default=0.65, help="Linear scaling factor.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence parameter.")
    parser.add_argument("--quad", type=float, default=0.0, help="Quadratic factor.")
    parser.add_argument("--repetition_penalty", type=float, default=4.0, help="Repetition penalty.")
    parser.add_argument("--repetition_penalty_window", type=int, default=3, help="Repetition penalty window.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling.")
    parser.add_argument("--progress_bar", default=True, action="store_true", help="Show progress bar.")
    
    args = parser.parse_args()

    print("Loading Zonos model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    # model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device) # only GPU
    
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