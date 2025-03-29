import argparse
import os
import torch
import torchaudio
import re
import math
import json
import sys # For sys.exit
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.conditioning import phonemize
from zonos.utils import DEFAULT_DEVICE as device
import logging

# Enable all DEBUG logging:
logging.basicConfig(level=logging.DEBUG)
# To enable also phonemizer DEBUG logging:
# logging.getLogger("phonemizer").setLevel(logging.DEBUG)

# --- Constants for Timing Logic ---
DEFAULT_END_BUFFER_S = 0.15         # Buffer time before next segment
DEFAULT_MAX_DURATION_FACTOR = 2.0   # Max stretch factor for duration
MIN_CALCULATION_DURATION_S = 0.01   # Avoid division by zero

def time_to_seconds(time_str: str) -> float:
    """Converts time from HH:MM:SS,mmm or HH:MM:SS.mmm format to seconds."""
    # Allow comma or dot separator for milliseconds
    match = re.match(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d+)", time_str)
    if not match:
        raise ValueError(f"Invalid time format: {time_str}")
    hours, minutes, seconds, ms_str = match.groups()
    ms = int(ms_str.ljust(3, '0')[:3])
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + ms / 1000.0

def parse_srt(srt_path: str) -> list:
    """Parses an SRT file using the original regex approach."""
    segments = []
    try:
        with open(srt_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # Use original regex, slightly adapted lookahead for EOF robustness
        srt_pattern = re.compile(r"""(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)\n*(?=\d+\n\d{2}:\d{2}:\d{2},\d{3}|\Z)""", re.DOTALL | re.MULTILINE)
        for match in srt_pattern.finditer(content):
            try:
                segments.append({
                    'index': int(match.group(1)),
                    'start_time': time_to_seconds(match.group(2)),
                    'end_time': time_to_seconds(match.group(3)),
                    'text': match.group(4).strip().replace('\n', ' ') # Clean text
                })
            except ValueError as e:
                logging.warning(f"Skipping segment {match.group(1)} due to time parse error: {e}")
    except FileNotFoundError:
        logging.error(f"SRT file not found: {srt_path}")
    except Exception as e:
        logging.error(f"Error reading/parsing SRT {srt_path}: {e}")
    segments.sort(key=lambda x: x['index']) # Ensure order
    return segments

def preprocess_audio(file_path: str, model: Zonos) -> torch.Tensor | None:
    """Loads and preprocesses audio prefix (e.g., silence). Returns codes or None."""
    try:
        wav, sr = torchaudio.load(file_path)
        wav = wav.mean(0, keepdim=True) # Ensure mono
        wav = model.autoencoder.preprocess(wav, sr).to(device, dtype=torch.float32)
        return model.autoencoder.encode(wav.unsqueeze(0))
    except Exception as e:
        logging.error(f"Failed to preprocess audio {file_path}: {e}")
        return None

def generate_audio(text: str, language: str, model: Zonos, speaker_embedding: torch.Tensor,
                   audio_prefix_codes: torch.Tensor, output_path: str, speaking_rate: float) -> bool:
    """Generates audio using Zonos and saves as WAV. Returns True on success."""
    print(f"    Generating audio for text: {text[:60]}...") # Keep user's print
    logging.info(f"  Synthesizing: '{text[:60]}...' (Rate: {speaking_rate:.1f}) -> {os.path.basename(output_path)}")
    torch.manual_seed(423)
    cond_dict = make_cond_dict(text=text,
                               speaker=speaker_embedding,
                               language=language,
                               speaking_rate=speaking_rate)
    conditioning = model.prepare_conditioning(cond_dict)
    codes = model.generate(prefix_conditioning=conditioning, audio_prefix_codes=audio_prefix_codes)
    model.autoencoder.save_codes(output_path, codes)

    logging.info(f"  Audio saved: {os.path.basename(output_path)}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate audio snippets from SRT, adjusting rate based on available time.")
    parser.add_argument("--language", required=True, help="Language code (e.g., 'en').")
    parser.add_argument("--srt", required=True, help="Path to the segmented SRT file.")
    parser.add_argument("--output", default="output_audio", help="Output directory for WAV and JSON.")
    parser.add_argument("--reference_audio", required=True, help="Path to reference audio for voice cloning.")

    parser.add_argument("--target_rate", type=float, default=15.0, help="Target speaking rate (model units).")
    parser.add_argument("--buffer", type=float, default=DEFAULT_END_BUFFER_S, help="Buffer time (s) before next segment.")
    parser.add_argument("--max_duration_factor", type=float, default=DEFAULT_MAX_DURATION_FACTOR, help="Max stretch factor for duration.")
    parser.add_argument("--force_regen", action='store_true', help="Regenerate all files.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- Load Model and Reference Audio ---
    print("Loading Zonos model...")
    try:
        model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        print("  -> Model loaded.")
        print("Loading reference audio...")
        wav, sr = torchaudio.load(args.reference_audio)
        speaker_embedding = model.make_speaker_embedding(wav, sr)
        print("  -> Speaker embedding generated.")
    except Exception as e:
        logging.error(f"Fatal: Failed to load model or reference audio: {e}", exc_info=True)
        sys.exit(1)

    # --- Parse SRT and Load Silence ---
    print("Parsing SRT file...")
    segments = parse_srt(args.srt)
    if not segments: sys.exit("Fatal: No segments parsed from SRT.")
    print(f"  -> Parsed {len(segments)} segments.")
    try:
        srt_mtime = os.path.getmtime(args.srt)
    except FileNotFoundError: sys.exit("Fatal: SRT file disappeared after parsing.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    silence_audio_path = os.path.join(script_dir, "assets", "silence_100ms.wav")
    silence_audio_codes = preprocess_audio(silence_audio_path, model)
    if silence_audio_codes is None: sys.exit("Fatal: Failed to load silence prefix.")

    # --- Process Segments ---
    total_segments = len(segments)
    for i, segment in enumerate(segments):
        segment_idx_str = f"{segment['index']:04d}"
        output_filename = os.path.join(args.output, f"{segment_idx_str}.wav")
        output_meta_filename = os.path.join(args.output, f"{segment_idx_str}.json")

        # Check cache
        if not args.force_regen and os.path.exists(output_filename) and os.path.getmtime(output_filename) >= srt_mtime:
            print(f" - Skipping {os.path.basename(output_filename)} (up-to-date)")
            continue

        logging.info(f"\nProcessing Segment {i+1}/{total_segments} (Index: {segment['index']}) '{segment['text'][:40]}...'")

        # --- Calculate Available Time for Speech ---
        original_duration = max(MIN_CALCULATION_DURATION_S, segment['end_time'] - segment['start_time'])
        if i < total_segments - 1: # Not the last segment
            available_duration = (segments[i+1]['start_time'] - segment['start_time']) - args.buffer
        else: # Last segment
            available_duration = original_duration
        # Apply max duration factor limit
        max_allowed_duration = original_duration * args.max_duration_factor
        final_available_duration = max(MIN_CALCULATION_DURATION_S, min(available_duration, max_allowed_duration))
        logging.debug(f"  Original Duration: {original_duration:.3f}s, Available (capped): {final_available_duration:.3f}s")

        # --- Calculate Required Speaking Rate ---
        try:
            segment_text = segment['text'].strip()
            if not segment_text: logging.warning(f"Segment {segment['index']} empty, skipping."); continue
            phonemes = phonemize([segment_text], [args.language])[0]
            if not phonemes: logging.warning(f"Phonemization failed for segment {segment['index']}, skipping."); continue
            num_phonemes = len(phonemes)
        except Exception as e:
            logging.error(f"Phonemization error for segment {segment['index']}: {e}", exc_info=True); continue

        required_rate_pps = num_phonemes / final_available_duration
        logging.debug(f"  Num Phonemes: {num_phonemes}, Required Rate: {required_rate_pps:.2f} pps")

        # --- Determine Final Speaking Rate (Model Units) ---
        # Use target rate unless required rate is higher (assuming higher value means faster speech)
        # This assumes args.target_rate is comparable to required_rate_pps or is the desired floor.
        final_model_speaking_rate = args.target_rate
        if required_rate_pps > args.target_rate:
            final_model_speaking_rate = required_rate_pps # Speed up
            # Keep original print message format if speed-up happens
            print(f" - Speaking rate must be increased from {args.target_rate:.1f} to {final_model_speaking_rate:.1f} (model units) for {num_phonemes} phonemes in {final_available_duration:.2f}s.")
        logging.info(f"  Final Rate: {final_model_speaking_rate:.2f} (model units)")

        # --- Save Metadata ---
        metadata = {
            "text": segment_text, 
            "phonemes": phonemes,
            "num_phonemes": num_phonemes,
            "duration": f"{final_available_duration:.3f}", # Key field
            "speaking_rate": f"{final_model_speaking_rate:.2f}",
            "language": args.language, 
            "output_filename": output_filename,
        }
        try:
            with open(output_meta_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2) # Use indent=2 for slightly less space
        except Exception as e:
            logging.error(f"Failed to write metadata for segment {segment['index']}: {e}")
            # Continue generation even if metadata fails? Or skip? Let's skip.
            continue

        # --- Generate Audio ---
        print(f" - Generating {output_filename}") # Keep user's print
        success = generate_audio(segment_text, args.language, model, speaker_embedding,
                                 silence_audio_codes, output_filename, final_model_speaking_rate)

        if not success:
            logging.warning(f"Generation failed for segment {segment['index']}. Cleaning up.")
            # Attempt cleanup
            if os.path.exists(output_filename): os.remove(output_filename)
            if os.path.exists(output_meta_filename): os.remove(output_meta_filename)


    print(" -> Audio snippets generation process completed.") # Keep user's print
    logging.info("--- Processing Finished ---")

if __name__ == "__main__":
    main()