import argparse
import os
import torch
import torchaudio
import re
import math
import json
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.conditioning import phonemize
from zonos.utils import DEFAULT_DEVICE as device
import logging

# To enable all DEBUG logging:
logging.basicConfig(level=logging.DEBUG)
# Or enable only phonemizer DEBUG logging:
logging.getLogger("phonemizer").setLevel(logging.DEBUG)

def time_to_seconds(time_str) -> float:
    """Converts time from HH:MM:SS,mmm format to seconds as a float."""
    hours, minutes, seconds, milliseconds = re.match(r"(\d+):(\d+):(\d+),(\d+)", time_str).groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000.0

def parse_srt(srt_path) -> list:
    """Parses an SRT file and returns a list of segments with their text and timestamps in seconds."""
    with open(srt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    segments = []
    srt_pattern = re.compile(r"""(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)\n*(?=\d+|$)""", re.DOTALL)
    
    for match in srt_pattern.finditer(content):
        segments.append({
            'index': int(match.group(1)),
            'start_time': time_to_seconds(match.group(2)),  # Convert to float seconds
            'end_time': time_to_seconds(match.group(3)),    # Convert to float seconds
            'text': match.group(4).replace('\n', ' ')
        })
    
    return segments

def preprocess_audio(file_path, model):
    """Loads and preprocesses audio for use as prefix."""
    wav, sr = torchaudio.load(file_path)
    wav = wav.mean(0, keepdim=True)
    wav = model.autoencoder.preprocess(wav, sr)
    wav = wav.to(device, dtype=torch.float32)
    return model.autoencoder.encode(wav.unsqueeze(0))

def generate_audio(text, language, model, speaker_embedding, audio_prefix_codes, output_path, speaking_rate=15.0):
    """Generates an audio snippet using the Zonos model with optional prefix audio."""
    print(f"    Generating audio for text: {text}")
    torch.manual_seed(423)
    
    # Add ellipsis to the text to get a more natural ending
    cond_dict = make_cond_dict(text=text + " ... ...", 
                               speaker=speaker_embedding,
                               language=language,
                               speaking_rate=speaking_rate)
    conditioning = model.prepare_conditioning(cond_dict)
    
    codes = model.generate(
      prefix_conditioning=conditioning, 
      audio_prefix_codes=audio_prefix_codes)
    
    model.autoencoder.save_codes(output_path, codes)

def main():
    parser = argparse.ArgumentParser(description="Generate audio snippets from an SRT file using Zonos.")
    parser.add_argument("--language", required=True, help="Language code for the text-to-speech synthesis.")
    parser.add_argument("--srt", required=True, help="Path to the SRT file.")
    parser.add_argument("--output", default="output_audio", help="Directory to save audio snippets.")
    parser.add_argument("--reference_audio", default="assets/exampleaudio.mp3", help="Path to an example speaker audio file.")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading Zonos model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    print("  -> Model loaded successfully.")
    
    print("Loading reference audio...")
    wav, sampling_rate = torchaudio.load(args.reference_audio)  # Provide an example speaker file
    speaker_embedding = model.make_speaker_embedding(wav, sampling_rate)
    print("  -> Speaker embedding generated.")
    
    print("Parsing SRT file...")
    segments = parse_srt(args.srt)
    srt_mtime = os.path.getmtime(args.srt)
    print(f"  -> Parsed {len(segments)} segments.")
    
    silence_audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "silence_100ms.wav")
    silence_audio_codes = preprocess_audio(silence_audio_path, model)
    previous_audio_codes = silence_audio_codes
    
    for i, segment in enumerate(segments):
        output_filename = os.path.join(args.output, f"{segment['index']:04}.wav")

        # Check if the WAV file exists and if it is newer than the SRT file
        if os.path.exists(output_filename) and os.path.getmtime(output_filename) >= srt_mtime:
            print(f" - Skipping {output_filename} (up-to-date)")
            continue  # Skip this segment if the audio file is already up-to-date

        # Determines number of phonemes in the text
        phonemes = phonemize([segment['text']], [args.language])[0]
        # phonemes = model.phonemizer.phonemize(segment['text'], language=args.language)

        # Determine if we need to increase speaking rate in phonemes per minute
        speaking_rate = 15.0

        duration = segment['end_time'] - segment['start_time']
        phonemes_per_minute = len(phonemes) / duration
        if phonemes_per_minute > speaking_rate:
            print(f" - Speaking rate must be increased from {speaking_rate} to {phonemes_per_minute:.1f} to utter {len(phonemes)} phonemes in {duration:.2f} seconds.")
            speaking_rate = math.ceil(phonemes_per_minute)

        # Write text and phonemes to text file
        output_txt_filename = os.path.join(args.output, f"{segment['index']:04}.json")
        with open(output_txt_filename, 'w', encoding='utf-8') as file:
            json_str = json.dumps({
                "text": segment['text'],
                "phonemes": phonemes,
                "number_of_phonemes": len(phonemes),
                "start_time": segment['start_time'],
                "end_time": segment['end_time'],
                "duration": duration,
                "speaking_rate": speaking_rate,                
                "language": args.language,
                "output_filename": output_filename,
            }, ensure_ascii=False, indent=4)
            file.write(json_str)

        # My experiments with using prefixing of the previous audio didn't show any positive results on audio quality
        # if False and i > 0 and (segments[i - 1]['end_time'] - segment['start_time']) > 0.01:
        #     prefix_audio_codes = previous_audio_codes
        #     text_to_generate = segments[i - 1]['text'] + " " + segment['text']  # Combine previous text
        #     print(f" - Generating (cont) {output_filename}")
        
        prefix_audio_codes = silence_audio_codes
        text_to_generate = segment['text']
        print(f" - Generating {output_filename}")

        generate_audio(text_to_generate, args.language, model, speaker_embedding, prefix_audio_codes, output_filename, speaking_rate)
    
    print(" -> Audio snippets generated successfully!")

if __name__ == "__main__":
    main()
