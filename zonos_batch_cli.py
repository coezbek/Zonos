import argparse
import json
import os
import torch
import math
import logging
import warnings
warnings.filterwarnings(action='ignore', message=r".*eprecated.*") # category=FutureWarning, 
import torchaudio
import torch.nn.functional as F

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

logger = logging.getLogger(__name__)

def load_audio(file_paths, model):
    """Loads and preprocesses multiple audio files, left-padding them to align to the right."""
    if isinstance(file_paths, str):
        file_paths = [file_paths]  # Convert single file path to list

    wavs = []
    max_length = 0  # Track max length to determine padding

    # Load and preprocess each file
    for file_path in file_paths:
        wav, sr = torchaudio.load(file_path)

        logger.debug(f"File: {file_path} | Shape: {wav.shape} | Sample Rate: {sr} | Channels: {wav.shape[0]}")

        if wav.shape[0] == 2:
            logger.warning(f"File: {file_path} | WARN: Prefix audio is stereo! Converting to mono")
            wav = wav.mean(0, keepdim=True)  # Convert to mono if needed
        
        if sr != 44_100:
            logger.warning(f"File: {file_path} | WARN: Prefix audio is {sr} Hz. Resampling to 44.1 kHz.")
            wav = torchaudio.functional.resample(wav, sr, 44_100)
        
        wavs.append(wav)

    max_length = max(math.ceil(w.shape[-1] / 512) * 512 for w in wavs)

    # Pad all waveforms to align them to the right
    padded_wavs = []
    for i, wav in enumerate(wavs):
        left_pad = max_length - wav.shape[-1]
        logger.debug(f"File: {file_paths[i]} | Left Padding: {left_pad} samples")  # Debugging info
        
        padded_wav = torch.nn.functional.pad(wav, (left_pad, 0), value=0)  # Left padding only
        padded_wavs.append(padded_wav)

    # Stack into a batch and encode
    batch_wav = torch.stack(padded_wavs).to(device, dtype=torch.float32)
    return model.autoencoder.encode(batch_wav)

def get_average_quality(input_aesthetics, quality_keys=['CE', 'CU', 'PQ']) -> float:
    if not isinstance(input_aesthetics, list):
        input_aesthetics = [input_aesthetics]

    values = [i[j] for i in input_aesthetics for j in quality_keys]
    return sum(values) / len(values)


def quality_string(input_aesthetics):
    """
    Converts a list of aesthetic scores into a formatted string with averages.
    Example output: "CE=4.2 CU=5.9 PC=2.1 PQ=6.8 AQ=4.8"
    """
    if not isinstance(input_aesthetics, list):
        input_aesthetics = [input_aesthetics]

    q = ['CE', 'CU', 'PQ'] # 'Production Quality PC' isn't really relevant here
    d = {j: sum(i[j] for i in input_aesthetics) / len(input_aesthetics) for j in q}

    return " ".join([f"{i}={d[i]:.1f}" for i in q] + [f"AQ={sum(d[k] for k in q)/len(q):.2f}", f"(n={len(input_aesthetics)})"])


def generate_audio(args, model, speaker_embedding, prefix_audio_codes, prefix_audio_text):
    """Generates speech for multiple texts in a batch."""

    if args.audio_aesthetics:
        from audiobox_aesthetics.infer import initialize_predictor
        predictor = initialize_predictor()

    # Create a dictionary to collect outputs per original text
    if args.bestof != 0:
        from collections import defaultdict
        text_to_outputs = defaultdict(list)

    # Apply text repetition
    args.text = [text for text in args.text for _ in range(args.text_repeat)]
    batch_size = len(args.text)

    # Expand prefix audio codes for batch size
    prefix_audio_codes = prefix_audio_codes.repeat(batch_size // prefix_audio_codes.shape[0], 1, 1)

    # Expand prefix audio text for batch size
    prefix_audio_text = prefix_audio_text * (batch_size // len(prefix_audio_text))

    text = [(prefix_audio_text[i] + " " if prefix_audio_text[i] else "") + args.text[i] for i in range(batch_size)]

    overall_quality = []

    # For each repetition
    for repeat in range(args.batch_repeat):

        torch.manual_seed(args.seed + repeat)
        logging.info(f"Batch {repeat + 1}/{args.batch_repeat} | Seed: {args.seed + repeat}")

        # Can process only max_per_batch samples at a time
        for chunk_i in range(0, len(text), args.max_per_batch):
        
            # Expand speaker embedding for batch size
            # speaker_embedding = speaker_embedding.expand(batch_size, speaker_embedding.size(1), speaker_embedding.size(2))
            # speaker_embedding = speaker_embedding.expand(batch_size, -1)

            # Create conditioning dictionaries for each text
            cond_dict = make_cond_dict(
                text=text[chunk_i:chunk_i + args.max_per_batch],
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
            # print ("cond_dict: ", cond_dict)

            prefix_conditioning = model.prepare_conditioning(cond_dict)

            # print ("conditioning.size(): ", prefix_conditioning.size())
            # print ("speaker_embedding.size(): ", speaker_embedding.size())
            # print ("prefix_audio_codes.size(): ", prefix_audio_codes.size())
        
            # Generate codes
            codes = model.generate(
                prefix_conditioning,
                audio_prefix_codes=prefix_audio_codes[chunk_i:chunk_i + args.max_per_batch],
                max_new_tokens=args.max_new_tokens,
                cfg_scale=args.cfg_scale,
                batch_size=len(text[chunk_i:chunk_i + args.max_per_batch]),
                disable_torch_compile=True, # When batching with these big tensors, torch.compile() makes things substantially slower
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

            batch_quality = []

            for i, code in enumerate(codes):

                i = chunk_i + i  # Adjust index for the original text list

                # Determine the padding lengths
                pad_i = len(str(len(args.text)-1))  # Number of digits in max_i
                if args.batch_repeat == 1:
                    output_file = f"{args.output.rstrip('.wav')}_{i:0{pad_i}d}.wav"
                else:
                    batch_then_text_repeat = False
                    pad_repeat = len(str(args.batch_repeat-1))  # Number of digits in max_repeat
                    if batch_then_text_repeat:
                        output_file = f"{args.output.rstrip('.wav')}_{repeat:0{pad_repeat}d}_{i:0{pad_i}d}.wav"
                    else:
                        output_file = f"{args.output.rstrip('.wav')}_{i:0{pad_i}d}_{repeat:0{pad_repeat}d}.wav"

                # Ensure directory exists
                if len(os.path.dirname(output_file)) > 0:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                # Save the decoded audio
                wav = model.autoencoder.codes_to_wavs(code)[0]

                sr = model.autoencoder.sampling_rate

                torchaudio.save(output_file, wav, sr)

                if args.audio_aesthetics:
                    d = predictor.forward([{"path": wav, "sample_rate": sr}])[0]
                    batch_quality.append(d)
                    overall_quality.append(d)

                    print(f"Generated audio saved to {output_file} | Audiobox Aesthetics: {quality_string(d)} | {text[i][:20]}...")

                    if args.bestof != 0:
                        text_to_outputs[text[i]].append((output_file, d))
                else:
                    print(f"Generated audio saved to {output_file} | {text[i][:20]}...")

        if args.audio_aesthetics and len(codes) > 1:
            print(f"Batch Audiobox Aesthetics:   {quality_string(batch_quality)}")

    if args.audio_aesthetics and args.batch_repeat > 1:
        print(f"Overall Audiobox Aesthetics: {quality_string(overall_quality)}")

    # After all files are generated, sort and print the best outputs for each text in order of quality metric
    if args.bestof != 0:
        for text, outputs in text_to_outputs.items():

            if len(outputs) <= 1:
                continue

            outputs.sort(key=lambda x: get_average_quality(x[1]), reverse=True)
            
            if args.bestof == -1:
                print("Outputs for '{text}':")
            else:
                print(f"Best {args.bestof} of {len(outputs)} outputs for '{text}':")
                outputs = outputs[:args.bestof] if len(outputs) > args.bestof else outputs

            for i, (output_file, d) in enumerate(outputs):
                print(f"  {i + 1:>2}: {output_file} | Audiobox Aesthetics: {quality_string(d)}")

def main():
    parser = argparse.ArgumentParser(description="Generate speech with Zonos CLI (Batch Mode).")
    parser.add_argument("--text", nargs="+", help="List of texts to generate speech for.")
    parser.add_argument("--text_file", default=None, help="Path to a text file with one text per line.")
    parser.add_argument("--language", default="en-us", help="Language code (e.g., en-us, de).")
    parser.add_argument("--reference_audio", default="assets/exampleaudio.mp3", help="Path to reference speaker audio.")
    parser.add_argument("--prefix_audio", "--audio_prefix", nargs="+", default=None, help="Path to prefix audio (default: 350ms silence).")
    parser.add_argument("--output", default="output.wav", help="Output wav file prefix.")
    parser.add_argument("--seed", type=int, default=423, help="Random seed for reproducibility.")
    parser.add_argument("--max_per_batch", type=int, default=-1, help="Max number of samples to generate at the same time. Each sample needs roughly 350mb of vram + 2GB for the models. 6 GB = max 12, 24 GB max 64 samples. Use -1 for best guess.")
    parser.add_argument("--batch_repeat", type=int, default=1, help="Number of times to repeat the entire batch generation (seed is incremented by 1).")
    parser.add_argument("--text_repeat", type=int, default=1, help="Number of times to repeat each text in the same batch.")
    parser.add_argument("--audio_aesthetics", action='store_true', help="Output audiobox-aesthetics per file.")
    parser.add_argument('--verbose', action='store_true', help="Print verbose output.")
    parser.add_argument('--verbose_sampling', action='store_true', help="Print verbose sampling output.")
    parser.add_argument("--bestof", "--best_of", type=int, default=0, help="Report the best n samples for each text sorted by quality. Use -1 for all samples. 0 to disable.")
 
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
    parser.add_argument("--repetition_penalty", type=float, default=2.5, help="Repetition penalty.")
    parser.add_argument("--repetition_penalty_window", type=int, default=8, help="Repetition penalty window.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature scaling.")
    parser.add_argument("--progress_bar", default=True, action="store_true", help="Show progress bar.")
    
    args = parser.parse_args()

    if (args.text_file is not None) and (args.text is not None):
        raise ValueError("Please provide either --text or --text_file, not both.")
    
    if args.text_file is not None:
        if not os.path.exists(args.text_file):
            raise FileNotFoundError(f"Text file '{args.text_file}' does not exist.")
        with open(args.text_file, "r", encoding="utf-8") as f:
            args.text = [line.strip() for line in f if line.strip()]

    if args.text is None:
        raise ValueError("Please provide --text or --text_file with texts to generate speech for.")
    
    # Need audio_aesthetics for bestof
    if args.bestof != 0:
        args.audio_aesthetics = True

    if args.verbose_sampling:
        args.verbose = True
        logging.getLogger("zonos.sampling").setLevel(logging.DEBUG)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        logging.getLogger("phonemizer").setLevel(logging.DEBUG)
        logging.getLogger("filelock").setLevel(logging.WARNING)
        logging.getLogger("audiobox_aesthetics").setLevel(logging.WARNING)
        logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.WARNING)

    if device.type == "cuda":
        
        vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # Convert to GB

        # Assume base model 4GB, 400 MB per sample
        estimated_max_per_batch = int(max(((vram - 4) // 0.400), 1)) # 4 is the minimum batch size, round to nearest 4
        
        if args.max_per_batch == -1:
            args.max_per_batch = estimated_max_per_batch
            logger.info(f"Detected VRAM: {vram:.2f} GB. Setting max_per_batch to {args.max_per_batch}.")
        else:
            if args.max_per_batch > estimated_max_per_batch:
                logger.warning(f"max_per_batch ({args.max_per_batch}) exceeds estimated max ({estimated_max_per_batch}) for {vram:.2f} GB of vRAM.")
            else:
                logger.info(f"max_per_batch set to {args.max_per_batch}, estimated max is {estimated_max_per_batch}.")

    else:
        if args.max_per_batch == -1:
            raise RuntimeError("CUDA is required for automatic batch size detection.")

    from pytictoc import TicToc
    t = TicToc()
    t.tic()

    print("Loading Zonos model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    # model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device) # only GPU
    t.toc("Loading complete in", restart=True)
    
    print("Loading speaker reference audio...")
    wav, sr = torchaudio.load(args.reference_audio)
    speaker_embedding = model.make_speaker_embedding(wav, sr)
    t.toc("Speaker embedding complete in", restart=True)
    
    print("Loading prefix audio...")
    if args.prefix_audio:
        prefix_audio_codes = load_audio(args.prefix_audio, model)

        # Load associated transcripts from transcript.json
        if os.path.exists("transcripts.json"):
            with open("transcripts.json", "r") as f:
                transcript = json.load(f)

                prefix_audio_text = []
                for prefix_audio in args.prefix_audio:
                    key = os.path.splitext(os.path.basename(prefix_audio))[0]
                    if key not in transcript:
                        print(f"⚠️  Warning: Key '{key}' not found in transcript.json. Using empty string as prefix text.")
                    prefix_audio_text.append(transcript.get(key, ""))
        else:
            prefix_audio_text = [""] * len(args.prefix_audio)
    else:
        silence_path = "assets/silence_100ms.wav"  # Ensure this file exists
        prefix_audio_codes = load_audio(silence_path, model)
        prefix_audio_text = [""]
    t.toc("Prefix audio complete in", restart=True)
    
    print("Generating speech...")
    generate_audio(args, model, speaker_embedding, prefix_audio_codes, prefix_audio_text)
    t.toc("Generating speech complete in", restart=True)
    
if __name__ == "__main__":
    main()