#!/usr/bin/env python
import os
import torch
import torchaudio
import logging

# To enable INFO logging:
# logging.basicConfig(level=logging.INFO)
# To enable DEBUG logging:
# logging.basicConfig(level=logging.DEBUG)
# Or enable only phonemizer DEBUG logging:
# logging.getLogger("phonemizer").setLevel(logging.DEBUG)

from zonos.utils import set_device
set_device("fastest")  # Set the device to the fastest available option
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
from zonos.speaker_utils import SpeakerUtils

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def keep_plot_alive():
    while True:
        plt.pause(0.1)  # Gives time to the GUI event loop
        if not plt.get_fignums():
            print("All plots closed. Stopping loop.")
            break

def compare_speaker_embeddings(vec1: torch.Tensor, vec2: torch.Tensor, name="speaker_embedding"):
    """
    Compare two speaker embeddings:
    - Computes cosine similarity and Euclidean distance
    - Plots both vectors
    - Plots absolute difference per dimension
    """
    vec1 = vec1.squeeze().to(dtype=torch.float32)
    vec2 = vec2.squeeze().to(dtype=torch.float32)
    assert vec1.shape == vec2.shape, f"Vectors must have the same shape"
    assert vec1.dim() == 1, f"Vectors must be 1D, but is {vec1.dim()}D with shape {vec1.shape}"

    # Normalize if needed (optional but common)
    vec1a = F.normalize(vec1, p=2, dim=0)
    vec2a = F.normalize(vec2, p=2, dim=0)

    # Similarity metrics
    cos_sim = F.cosine_similarity(vec1, vec2, dim=0).item()
    euclidean = torch.norm(vec1 - vec2, p=2).item()
    diff = torch.abs(vec1 - vec2).cpu().numpy()
    rel_diff = ((vec1 - vec2) / torch.max(torch.abs(vec1 - vec2))).cpu().numpy()

    # Convert for plotting
    v1 = vec1.cpu().numpy()
    v2 = vec2.cpu().numpy()

    # Plot 1: Vectors
    fig = plt.figure(figsize=(12, 4))
    plt.plot(v1, label='vec1')
    plt.plot(v2, label='vec2')
    plt.title(f"Speaker Embedding Comparison\nCosine Similarity: {cos_sim:.4f} | Euclidean Distance: {euclidean:.4f}")
    plt.xlabel("Dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(f"{name}-vec.png")

    # Plot 2: Absolute difference
    fig2 = plt.figure(figsize=(12, 3))
    plt.bar(np.arange(len(diff)), diff)
    plt.title("Absolute Difference per Dimension")
    plt.xlabel("Dimension")
    plt.ylabel("Abs Difference")
    plt.tight_layout()
    fig2.savefig(f"{name}-absolute-diff.png")
    # plt.show(block=False)

    fig3 = plt.figure(figsize=(12, 3))
    plt.bar(np.arange(len(rel_diff)), rel_diff)
    plt.title("Absolute Difference per Dimension")
    plt.xlabel("Dimension")
    plt.ylabel("Abs Difference")
    plt.tight_layout()
    fig3.savefig(f"{name}-relative-diff.png")

    return {
        "cosine_similarity": cos_sim,
        "euclidean_distance": euclidean,
        "abs_difference": diff
    }

def main(lang, reference_audio):
    # Load the Zonos model.
    print("Loading Zonos model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device) 
    print("  -> Model loaded.")
    
    # Load reference speaker audio to generate speaker embedding.
    print("Loading speaker embedding...")
    spk_utils = SpeakerUtils(model=model)

    # age: ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', 'prefer not to answer']
    # emotion: ['adoration', 'amazement', 'amusement', 'anger', 'confusion', 'contentment', 
    #           'cuteness', 'desire', 'disappointment', 'disgust', 'distress', 'embarassment', 
    #           'extasy', 'fear', 'guilt', 'interest', 'neutral', 'pain', 'pride', 'realization',
    #           'relief', 'sadness', 'serenity']
    # ethnicity: ['asian', 'black or african american', 'hispanic or latino', 'prefer not to answer', 'white or caucasian']
    # gender: ['female', 'male', 'non-binary / third gender', 'prefer not to answer']
    # height: ["5' - 5'3", "5'4 - 5'7", "5'8 - 5'11", "6' - 6'3", "< 5'", 'prefer not to answer']
    # native language: ['dari', 'de_de', 'en_gb', 'en_us', 'es', 'prefer not to answer', 'ru', 'ukrainian', 'zh']
    # reading_style: ['emotion', 'fast', 'highpitch', 'loud', 'lowpitch', 'regular', 'slow', 'whisper']
    # speaker_id: p001 - p107 
    # weight: ['100 - 120 lbs', '120 - 140 lbs', '140 - 160 lbs', '160 - 180 lbs', '180 - 200 lbs', '200 - 220 lbs', '220 - 240 lbs', '260 - 280 lbs', '280 - 300 lbs', '340 - 360 lbs', 'prefer not to answer']
    
    if reference_audio is None:
            
        if True:
            speaker_embedding = spk_utils.load_average({
                "gender": "female",
                "age": "66-75",
                "reading_style": "regular",
            })

            speaker_embedding = speaker_embedding * 0.8 + 0.4 * spk_utils.load_average({
                "gender": "female",
                "age": "66-75",
                "reading_style": "emotion",
                "emotion": "interest"
            })

        if False: # German accent
            speaker_embedding = spk_utils.load_average({
                "gender": "male",
                "native language": "de_de",
                "reading_style": "regular",
            })
        
        if False: # Disappointment
            speaker_embedding = spk_utils.load_average({
                "gender": "male", 
                # "reading_style": "regular",
                "emotion": "disappointment",
                "native language": "en_us",
                "weight": "180 - 200 lbs",
                "speaker_id": "p009",
            })
            speaker_embedding2 = spk_utils.load_average({
                "gender": "male", 
                "reading_style": "regular",
                "native language": "en_us",
                "weight": "180 - 200 lbs",
                "speaker_id": "p009",
            })

            speaker_embedding = speaker_embedding2 + 2 * (speaker_embedding - speaker_embedding2)

        if True:
            speaker_embedding = spk_utils.load_average({
                "gender": "male", 
                "reading_style": "regular",
                "native language": "en_us",
                "weight": "180 - 200 lbs",
                "speaker_id": "p009",
            })
    else:
        speaker_embedding = spk_utils.compute_average([spk_utils.get_speaker_embedding(r, force_recalc=True) for r in reference_audio])

    if False:
        speaker_embedding_original = spk_utils.get_speaker_embedding("01_reference.wav")
        speaker_embedding_with_silence = spk_utils.get_speaker_embedding("02_silence.wav")
        speaker_embedding_part_1 = spk_utils.get_speaker_embedding("01_silence_part1.wav")
        speaker_embedding_part_2 = spk_utils.get_speaker_embedding("01_silence_part2.wav")
        speaker_embedding_lufs = spk_utils.get_speaker_embedding("01_silence_LUFS_normalized.wav")
        
        speaker_embedding_avg = spk_utils.compute_average([speaker_embedding_part_1, speaker_embedding_part_2])

        speaker_embedding_us = spk_utils.load_average({
            "gender": "male", 
            "reading_style": "regular",
            "native language": "en_us",
            "weight": "180 - 200 lbs",
            "speaker_id": "p009",
        })

        compare_speaker_embeddings(speaker_embedding_original, speaker_embedding_with_silence)
        compare_speaker_embeddings(speaker_embedding_lufs, speaker_embedding_with_silence)
        
        compare_speaker_embeddings(speaker_embedding_with_silence, speaker_embedding_avg)
        compare_speaker_embeddings(speaker_embedding_original, speaker_embedding_avg)
        compare_speaker_embeddings(speaker_embedding_original, speaker_embedding_us)

    if True:
        speaker1 = spk_utils.get_speaker_embedding("german_mixin_voice.wav")
        speaker2 = spk_utils.get_speaker_embedding("german_mixin_voice3.wav")
        compare_speaker_embeddings(speaker1, speaker2, name="german_mixin_voice vs german_mixin_voice2")

        speaker_embedding = speaker1 + (speaker2 - speaker1) * 3.0


        
        
    print("  -> Speaker embedding generated.")

    # Load the prefix audio (silence).
    print("Loading prefix audio...")
    prefix_audio_path = os.path.join("assets", "silence_350ms.wav")
    prefix_audio_codes = model.autoencoder.load_prefix_audio(prefix_audio_path, device=device)
    print(" -> Prefix audio loaded.")
    
    
    print(f"Language: {lang}")

    sentences_to_generate = 4
    sentences = [SpeakerUtils.random_sentence(lang=lang) for _ in range(sentences_to_generate)]
    print (f" -> Random sentence(s): {sentences}")

    # Create and explicitly set the conditioning dictionary.
    cond_dict = make_cond_dict(
        text=sentences, # Hello from Zonos, the state of the art text to speech model.",
        speaker=speaker_embedding,
        language=lang,
        emotion=[1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2], # [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral]
        fmax=22050.0,          # 44.1 kHz output frequency (also support 24000 for 48 kHz)
        pitch_std=45.0,        # Standard deviation for pitch variation
        speaking_rate=15.0,    # Speaking rate in phonemes per minute
        vqscore_8=[0.78] * 8,  # Target VoiceQualityScore for each 1/8th of the audio
        ctc_loss=0.0,          # CTC target loss
        dnsmos_ovrl=4.0,       # Overall DNSMOS score
        speaker_noised=False,  # Speaker noising disabled
        unconditional_keys={"emotion", "vqscore_8", "dnsmos_ovrl"}
    )
    
    # Prepare conditioning for the model.
    conditioning = model.prepare_conditioning(cond_dict)
    
    # Define generation parameters.
    generation_params = {
        "max_new_tokens": 86 * 30,
        "cfg_scale": 2.0,
        "batch_size": sentences_to_generate,
        "sampling_params": {
            "top_p": 0, 
            "top_k": 0,
            "min_p": 0,
            "linear": 0.8,
            "conf": 0.2,
            "quad": 0.0,
            "repetition_penalty": 1.3,
            "repetition_penalty_window": 16,
            "temperature": 1.0,
        },
        "progress_bar": True,
        "disable_torch_compile": True,
    }
    
    # For reproducibility.
    torch.manual_seed(421)
    
    # Generate audio codes with the provided prefix audio and generation parameters.
    codes = model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=prefix_audio_codes,
        **generation_params
    )

    if sentences_to_generate == 1:
        paths = ["sample_voice_cloning.wav"]
    else:
        paths = [f"sample_voice_cloning_{i}.wav" for i in range(sentences_to_generate)]    

    # Decode the generated codes and save
    model.autoencoder.save_codes(paths, codes)
    print(f"Audio saved to {paths}")

    # keep_plot_alive()

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Generate some sample audio snippets from the given reference audio or via voicehash from .voices")
    parser.add_argument("--language", default='en_us', help="Language code (e.g., 'en').")
    parser.add_argument("--reference_audio", nargs="+", default=None, help="Path to reference audio for voice cloning (or hash or directory).")
    parser.add_argument("--plot_speaker_embeddings", action="store_true", help="Plot speaker embeddings.")

    args = parser.parse_args()

    main(lang=args.language, reference_audio=args.reference_audio)
