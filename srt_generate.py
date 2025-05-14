import argparse
import os
import torch
import torchaudio
import glob
import re
import math
import string
import json
import sys # For sys.exit

from zonos.utils import set_device
set_device("memory") # Set device to one with most memory available GPU or CPU
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.conditioning import phonemize
from zonos.utils import DEFAULT_DEVICE as device
import logging

# Enable all DEBUG logging:
logging.basicConfig(level=logging.DEBUG)
# To enable also phonemizer DEBUG logging:
logging.getLogger("phonemizer").setLevel(logging.DEBUG)
# To enable detailled information about sampling:
logging.getLogger("zonos.sampling").setLevel(logging.DEBUG)
# logging.getLogger("zonos.sampling.trace").setLevel(logging.DEBUG)
logger = logging.getLogger(__file__)

# --- Constants for Timing Logic ---
DEFAULT_END_BUFFER_S = 0.01         # Buffer time before next segment
DEFAULT_MAX_DURATION_FACTOR = 2.0   # Max stretch factor for duration if there is sufficient time to next segment.
MIN_CALCULATION_DURATION_S = 0.01   # Avoid division by zero
MODEL_SAMPLING_RATE = 44100         # Zonos output sampling rate

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
                   audio_prefix_codes: torch.Tensor, output_path: str, speaking_rate: float,
                   final_available_duration: float, phonemes: str = None) -> tuple[bool, dict]:
    """Generates audio using Zonos and saves as WAV. Returns (True on success, candidate_qualities dict)."""
    print(f"    Generating audio for text: '{text}'{f" '{phonemes}'" if phonemes else ''}")
    batch_size = 16 # Trying to find the best audio from 16 candidates - Needs roughly 12 GB of VRAM
    
    # Create and explicitly set the conditioning dictionary.
    # Note: make_cond_dict initializes with basic keys.
    cond_dict = make_cond_dict(
        text=[text for _ in range(batch_size)],
        speaker=speaker_embedding,
        language=language,
        emotion=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2], # [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]
        fmax=22050.0,          # 44.1 kHz output frequency (also support 24000 for 48 kHz)
        pitch_std=25.0,        # Standard deviation for pitch variation
        speaking_rate=speaking_rate,    # Speaking rate in phonemes per minute
        vqscore_8=[0.78] * 8,  # Target VoiceQualityScore for each 1/8th of the audio
        ctc_loss=0.0,          # CTC target loss
        dnsmos_ovrl=4.0,       # Overall DNSMOS score
        speaker_noised=False,  # Speaker noising disabled
        unconditional_keys={"emotion", "vqscore_8", "dnsmos_ovrl"}
    )
    
    # Prepare conditioning for the model.
    conditioning = model.prepare_conditioning(cond_dict)

    linear = 0.72
    quad = (1/3)-(linear*4/15)
    conf = -0.5*quad

    # linear = 0.8
    # quad = 0.0
    # conf = 0.2

    # If the number of words is small, we don't need variability in sampling, we just need the most relevant output
    #if len(text.split()) == 1:
    top_k = 3
    top_p = 0.95
    linear = 0 # disable linear sampling
    #else:
    #    top_k = 0
    #    top_p = 0.999
        
    if logger.isEnabledFor(logging.DEBUG):
        print(f"  -> Linear: {linear:.3f}, Conf: {conf:.3f}, Quad: {quad:.3f}, Top P: {top_p:.3f}, Top K: {top_k}")

        if logging.getLogger("zonos.sampling.trace").isEnabledFor(logging.DEBUG):
            from zonos.sampling import print_unified_sampler_explanation
            print_unified_sampler_explanation(linear=linear, conf=conf, quad=quad)
            
    # Define generation parameters.
    generation_params = {
        "max_new_tokens": 86 * 30,
        "cfg_scale": 2.0,
        "batch_size": batch_size,
        "sampling_params": {
            "top_p": top_p, 
            "top_k": top_k,
            "min_p": 0.001,
            "linear": linear,
            "conf": conf,
            "quad": quad,
            "repetition_penalty": 1.1,
            "repetition_penalty_window": 8,
            "temperature": 1.0,
        },
        "progress_bar": True,
        "disable_torch_compile": True,
    }

    # Expand prefix audio codes for batch size
    audio_prefix_codes = audio_prefix_codes.repeat(batch_size // audio_prefix_codes.shape[0], 1, 1)
    
    print ("conditioning.size(): ", conditioning.size())
    print ("speaker_embedding.size(): ", speaker_embedding.size())
    print ("prefix_audio_codes.size(): ", audio_prefix_codes.size())
    
    # Generate audio codes with the provided prefix audio and generation parameters.
    codes = model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        **generation_params
    )

    wavs = model.autoencoder.codes_to_wavs(codes)

    filter_candidates = True
    # Remove the shortest and longest generated audio because facebook prefers longer audio
    if filter_candidates:
        num_to_filter = int(math.log2(len(wavs))) - 1
        logging.debug(f"  Filtering {num_to_filter} shortest and longest from {len(wavs)} wavs")
        if num_to_filter > 0:
            wavs_by_length = sorted(wavs, key=lambda x: x.shape[1])
            wavs = wavs_by_length[num_to_filter // 2:-num_to_filter]

    # Calculate quality
    quality_scores = model.autoencoder.audio_quality(wavs, MODEL_SAMPLING_RATE, average_overall=False)

    # Sort wavs by Average Quality (AQ) descending
    wavs_by_quality = sorted(zip(wavs, quality_scores), key=lambda x: x[1].get('AQ', 0.0), reverse=True)
        
    candidate_qualities_list = []
    save_candidates = True
    if save_candidates:
        # Delete existing files
        to_delete = f"{output_path[:-4]}_*.wav"
       
        for file in glob.glob(to_delete):
            try:
                os.remove(file)
            except Exception as e:
                logging.warning(f"  Failed to delete existing candidate file {file}: {e}")

        for i in range(len(wavs_by_quality)):

            wav, quality_score = wavs_by_quality[i]

            # Save candidate waves as {output_path}_1.wav, _2.wav, etc.
            candidate_output_path = f"{output_path[:-4]}_{i+1}.wav"
            torchaudio.save(candidate_output_path, wav, MODEL_SAMPLING_RATE)

            quality_string = model.autoencoder.quality_string(quality_score)
            
            selected_duration = wav.shape[1] / float(MODEL_SAMPLING_RATE)
            if selected_duration > final_available_duration:
                duration_warning = f"- WARN: exceeds {final_available_duration:.2f}s"
            else:
                duration_warning = ""
            logging.info(f" Candidate Audio saved: {os.path.basename(candidate_output_path)}: {quality_string} - Length: {selected_duration:.2f}s{duration_warning}")

            # Store path, scores, and string for sorting later
            candidate_qualities_list.append({
                "path": os.path.basename(candidate_output_path),
                "scores": quality_score,
                "quality_string": quality_string + f" Len: {selected_duration:.2f}s",
            }) 

    # Keep only the best audio from the generated samples
    wav, _ = wavs_by_quality[0]

    # Check duration before saving the final file
    selected_duration = wav.shape[1] / float(MODEL_SAMPLING_RATE)
    if selected_duration > final_available_duration:
        logging.warning(f"Selected audio duration ({selected_duration:.3f}s) exceeds available duration ({final_available_duration:.3f}s) for {os.path.basename(output_path)}")

    torchaudio.save(output_path, wav, MODEL_SAMPLING_RATE)

    logging.info(f"  Audio saved: {output_path}")

    # Convert list of dicts to dict for JSON {filename: quality_string}
    sorted_candidate_dict = {item["path"]: item["quality_string"] for item in candidate_qualities_list}

    return True, sorted_candidate_dict

def main():
    parser = argparse.ArgumentParser(description="Generate audio snippets from an SRT subtitle file, adjusting speaking rate based on available time.")
    parser.add_argument("--language", required=True, help="Language code (e.g., 'en').")
    parser.add_argument("--srt", required=True, help="Path to the segmented SRT file.")
    parser.add_argument("--output", default="output_audio", help="Output directory for WAV and JSON.")
    parser.add_argument("--reference-audio", nargs="+", required=True, help="Path to reference audio for voice cloning.")

    parser.add_argument("--target-rate", type=float, default=15.0, help="Target speaking rate (phonemes per minute).")
    parser.add_argument("--buffer", type=float, default=DEFAULT_END_BUFFER_S, help="Buffer time (s) before next segment.")
    parser.add_argument("--force-regen", action='store_true', help="Regenerate all files.")

    parser.add_argument("--seed", type=int, default=-1, help="Random seed for reproducibility.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # For reproducibility.
    if args.seed >= 0:
        print(f"Setting random seed to {args.seed}")
        torch.manual_seed(args.seed)
    else:
        import time
        seed = int(time.time())
        print(f"No seed provided, using generated seed: {seed}")
        torch.manual_seed(seed)

    model = None

    # --- Parse SRT and Load Silence ---
    print("Parsing SRT file...")
    segments = parse_srt(args.srt)
    if not segments: sys.exit("Fatal: No segments parsed from SRT.")
    print(f"  -> Parsed {len(segments)} segments.")
    try:
        srt_mtime = os.path.getmtime(args.srt)
    except FileNotFoundError: sys.exit("Fatal: SRT file disappeared after parsing.")

    # --- Process Segments ---
    total_segments = len(segments)
    for i, segment in enumerate(segments):
        segment_idx_str = f"{segment['index']:04d}"
        output_filename = os.path.join(args.output, f"{segment_idx_str}.wav")
        output_meta_filename = os.path.join(args.output, f"{segment_idx_str}.json")
        
        # --- Check Regeneration Conditions ---
        needs_regen = args.force_regen
        meta_exists = os.path.exists(output_meta_filename)
        output_exists = os.path.exists(output_filename)
        meta_mtime = os.path.getmtime(output_meta_filename) if meta_exists else 0
        output_mtime = os.path.getmtime(output_filename) if output_exists else 0

        if meta_mtime == 0 and output_mtime > 0:
            meta_mtime = output_mtime # If meta doesn't exist, use output mtime

        # Trigger regen if output doesn't exist or SRT is newer than output
        if not output_exists or output_mtime < srt_mtime:
            needs_regen = True
            reason = "output missing or older than SRT"

        # Trigger regen if meta doesn't exist or SRT is newer than meta
        elif not meta_exists:
            needs_regen = True
            reason = "meta missing"
        
        elif meta_mtime < srt_mtime:
            needs_regen = True
            reason = f"meta older than SRT: {time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(meta_mtime))} < {time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(srt_mtime))}"

        # Trigger regen if meta is newer than output (implies manual edit or failed previous run)
        # AND meta is newer than SRT (to ensure meta reflects latest SRT)
        elif meta_exists and output_exists and meta_mtime > output_mtime:
            needs_regen = True           
            reason = f"meta.json is {meta_mtime - output_mtime:.3f}s newer than output .wav"

        elif meta_exists and output_exists and meta_mtime > output_mtime and meta_mtime <= srt_mtime:
            # Meta is newer than output, but older than SRT -> Regen needed
            needs_regen = True
            reason = "meta newer than output but older than SRT"

        else:
            reason = ''

        logging.info(f"\nProcessing Segment {i+1}/{total_segments} (Index: {segment['index']}) '{segment['text']}'")

        # --- Initialize variables, potentially load from metadata ---
        segment_text = segment['text'].strip()
        language = args.language
        phonemes = None
        num_units = None
        final_model_speaking_rate = args.target_rate # Default
        metadata_loaded = False
        candidate_qualities = {} # Initialize empty
        metadata = {} # Ensure metadata is a dict

        # Load metadata BEFORE calculations if it exists and is newer than BOTH SRT and WAV
        # This allows overriding values before potentially recalculating them.
        if meta_exists and output_exists and meta_mtime >= srt_mtime and meta_mtime >= output_mtime and not args.force_regen:
            try:
                with open(output_meta_filename, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                logging.debug(f"  Loading data from modified metadata file: {output_meta_filename}")

                # Overwrite variables if they exist in the metadata
                segment_text = metadata.get("text", segment_text)
                phonemes = metadata.get("phonemes", None) # Load if present
                # Load speaking rate only if it's a valid float
                try:
                    loaded_rate = float(metadata.get("speaking_rate", final_model_speaking_rate))
                    final_model_speaking_rate = loaded_rate
                except (ValueError, TypeError):
                    logging.warning(f"  Invalid or missing 'speaking_rate' in metadata, will use default or recalculate.")
                    # Keep default final_model_speaking_rate
                language = metadata.get("language", language)
                # num_units will be recalculated if phonemes were loaded, or loaded if present (though less common)
                num_units = metadata.get("num_phonemes", None) # Load if present
                metadata_loaded = True # Flag that we loaded potentially overriding values
            except Exception as e:
                logging.warning(f"  Could not load or parse existing metadata {output_meta_filename}: {e}. Will recalculate if needed.")
                metadata = {} # Reset metadata if loading failed

        # --- Calculate Available Time for Speech (Always needed) ---
        original_duration = max(MIN_CALCULATION_DURATION_S, segment['end_time'] - segment['start_time'])
        if i < total_segments - 1: # Not the last segment
            available_duration = (segments[i+1]['start_time'] - segment['start_time']) - args.buffer
        else: # Last segment
            available_duration = original_duration
            
        # Apply max duration factor limit
        max_allowed_duration = original_duration * DEFAULT_MAX_DURATION_FACTOR
        final_available_duration = max(MIN_CALCULATION_DURATION_S, min(available_duration, max_allowed_duration))

        # --- Determine if calculations are needed ---
        # Need to calculate if forced, or if metadata wasn't loaded, or if key values are missing from loaded metadata
        recalculate_rate = True # Assume recalculation is needed
        if metadata_loaded and phonemes is not None and final_model_speaking_rate is not None:
             logging.debug("  Using phonemes and speaking rate from metadata.")
             recalculate_rate = False # Skip rate calculation if loaded and valid
             # Still need num_units if not loaded from metadata
             if num_units is None and phonemes:
                 non_phoneme_chars = set(string.punctuation) | {' ', 'ˈ', 'ˌ'}
                 num_units = sum(1 for char in phonemes if char not in non_phoneme_chars)
                 logging.debug(f"  Calculated num_units ({num_units}) from loaded phonemes.")
             elif num_units is not None:
                 logging.debug(f"  Using num_units ({num_units}) from metadata.")
             else: # phonemes is None or empty
                 logging.warning("  Metadata loaded but phonemes missing/empty. Will attempt recalculation.")
                 recalculate_rate = True # Force recalculation

        # Decide if actual regeneration (running the model) is needed
        # Regen is needed if forced, or if initial checks said so, OR if we need to recalculate rate
        if needs_regen:
            print(f" - Processing {os.path.basename(output_filename)} (Reason: {'force_regen' if args.force_regen else reason})")
        else:
            # If we loaded everything from up-to-date metadata and no recalc needed
            print(f" - Skipping {os.path.basename(output_filename)} (up-to-date based on metadata)")
            continue

        # --- Calculate Phonemes and Required Speaking Rate (only if needed) ---
        if recalculate_rate:
            logging.debug(f"  Original Duration: {original_duration:.3f}s, Available (capped): {final_available_duration:.3f}s")
            try:
                # Uppercase first letter of the string:
                if segment_text:
                    segment_text = segment_text[0].upper() + segment_text[1:]
                else:
                    logging.warning(f"Segment {segment['index']} text is empty, skipping generation."); continue

                if phonemes is None: # Only phonemize if not loaded
                    phonemes = phonemize([segment_text], [language])[0]
                    if not phonemes: logging.warning(f"Phonemization failed for segment {segment['index']}, skipping generation."); continue
                    logging.debug(f"  Phonemized: '{phonemes}'")

                # --- Select the counting method ---
                rate_count_method_to_use = 'phonemes_spaces_no_punct' # <--- SET METHOD HERE
                count_unit_type = "unknown"

                match rate_count_method_to_use:
                    case 'chars':
                        num_units = len(phonemes)
                        count_unit_type = "characters"
                    case 'phonemes_spaces_no_punct':
                        punctuation_to_exclude = set(string.punctuation) | {'ˈ', 'ˌ'}
                        num_units = sum(1 for char in phonemes if char not in punctuation_to_exclude)
                        count_unit_type = "phonemes+spaces (no punct)"
                    case 'phonemes_only':
                        non_phoneme_chars = set(string.punctuation) | {' ', 'ˈ', 'ˌ'}
                        num_units = sum(1 for char in phonemes if char not in non_phoneme_chars)
                        count_unit_type = "phonemes only (no spaces, no punct)"
                    case _:
                        logging.warning(f"Invalid internal setting '{rate_count_method_to_use}' for rate counting. Falling back to 'chars'.")
                        num_units = len(phonemes)
                        count_unit_type = "characters (fallback)"
                        rate_count_method_to_use = 'chars'

                logging.debug(f"  Count Method Used: '{rate_count_method_to_use}', Counted Units ({count_unit_type}): {num_units}")

                if num_units <= 0:
                     logging.warning(f"Calculated 0 effective units for rate calculation in segment {segment['index']} using method '{rate_count_method_to_use}'. Using 1 unit to avoid calculation errors.")
                     num_units = 1

            except Exception as e:
                logging.error(f"Phonemization or unit counting error for segment {segment['index']}: {e}", exc_info=True); continue

            required_rate_pps = math.ceil(num_units / final_available_duration)
            logging.debug(f"  Num {count_unit_type}: {num_units}, Required Rate: {required_rate_pps:.2f} pps")

            if required_rate_pps > 40.0:
                logging.warning(f"Warning: Required rate {required_rate_pps:.2f} pps exceeds 40.0 pps, capping.")
                required_rate_pps = 40.0

            # --- Determine Final Speaking Rate (Model Units) ---
            # Start with default/loaded rate, only increase if required rate is higher
            current_rate = final_model_speaking_rate # Use potentially loaded rate as base
            if required_rate_pps > current_rate:
                final_model_speaking_rate = required_rate_pps
                print(f" - Speaking rate must be increased from {current_rate:.1f} to {final_model_speaking_rate:.1f} (model units) for {num_units} {count_unit_type} in {final_available_duration:.2f}s.")
            else:
                # Keep the loaded or default rate if it's sufficient
                final_model_speaking_rate = current_rate
            logging.debug(f"  Final Calculated Rate: {final_model_speaking_rate:.2f} (model units)")
        # End of recalculate_rate block
        else:
             # Rate was loaded and deemed sufficient
             logging.debug(f"  Using loaded rate: {final_model_speaking_rate:.2f} (model units)")
             # Ensure num_units is valid if it was loaded
             if num_units is None:
                 logging.error(f"Logic error: Rate loaded but num_units is None for segment {segment['index']}. Skipping.")
                 continue


        # --- Load Model and Reference Audio opportunistically if we need to regen anything ---
        if model is None: # Only load model once if needed for any segment

            print("Loading Zonos model...")
            try:
                model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
                print("  -> Model loaded.")

                print("Loading reference audio...")
                if True:
                    from zonos.speaker_utils import SpeakerUtils
                    spk_utils = SpeakerUtils(model=model)

                    # args.reference_audio = args.reference_audio + ["/home/coezbek/2025/zonos/german_mixin_voice.wav"]
                    speaker_embedding = spk_utils.compute_average([spk_utils.get_speaker_embedding(r) for r in args.reference_audio])

                else:
                    from zonos.speaker_utils import SpeakerUtils
                    spk_utils = SpeakerUtils(model=model)
                    speaker_embedding = spk_utils.load_average({
                        "gender": "male", 
                        "age": "46-55",
                        "reading_style": "regular",
                        "native language": "en_us",
                        #"weight": "180 - 200 lbs",
                        #"speaker_id": "p009",
                    })
                    speaker_embedding = speaker_embedding * 0.8 + 0.4 * spk_utils.load_average({
                        "gender": "male", 
                        "age": "46-55",
                        "reading_style": "emotion",
                        "emotion": "interest",
                        "native language": "en_us",
                        #"weight": "180 - 200 lbs",
                        #"speaker_id": "p009",
                    })

                print("  -> Speaker embedding generated.")
            except Exception as e:
                logging.error(f"Fatal: Failed to load model or reference audio: {e}", exc_info=True)
                sys.exit(1)

            script_dir = os.path.dirname(os.path.abspath(__file__))
            silence_audio_path = os.path.join(script_dir, "assets", "silence_100ms.wav")
            silence_audio_codes = preprocess_audio(silence_audio_path, model)
            if silence_audio_codes is None: sys.exit("Fatal: Failed to load silence prefix.")

        # --- Generate Audio ---
        print(f" - Generating {output_filename}") # Keep user's print
        success, candidate_qualities = generate_audio(
            segment_text, language, model, speaker_embedding,
            silence_audio_codes, output_filename, final_model_speaking_rate,
            final_available_duration, phonemes=phonemes
        )

        # --- Save Metadata (always save after successful generation) ---
        if success:
            metadata_to_save = {
                "text": segment_text,
                "phonemes": phonemes,
                "num_phonemes": num_units, # Store the counted units (could be loaded or calculated)
                "duration": f"{final_available_duration:.3f}", # Available duration
                "speaking_rate": f"{final_model_speaking_rate:.2f}",
                "language": language,
                "output_filename": os.path.basename(output_filename),
                "candidates": candidate_qualities # Add sorted candidate quality strings
            }
            try:
                with open(output_meta_filename, 'w', encoding='utf-8') as f:
                    json.dump(metadata_to_save, f, ensure_ascii=False, indent=2)

                # Update the modification time of the metadata file to match the output file
                output_mtime = os.path.getmtime(output_filename)
                os.utime(output_meta_filename, (output_mtime, output_mtime))

                logging.debug(f"  Metadata saved to {output_meta_filename}")
            except Exception as e:
                logging.error(f"Failed to write metadata for segment {segment['index']} after generation: {e}")
                # Generation succeeded, but metadata failed. Keep audio? Yes.
        elif not success:
            logging.warning(f"Generation failed for segment {segment['index']}.")


    print(" -> Audio snippets generation process completed.") # Keep user's print
    logging.info("--- Processing Finished ---")

if __name__ == "__main__":
    main()