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

def main():
    # Load the Zonos model.
    print("Loading Zonos model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device) 
    print("  -> Model loaded.")
    
    # Load reference speaker audio to generate speaker embedding.
    print("Loading speaker embedding...")
    spk_utils = SpeakerUtils(model=model)

    # age: ['18-25', '26-35', '36-45', '46-55', '56-65', '66-75', 'prefer not to answer']
    # emotion: ['adoration', 'amazement', 'amusement', 'anger', 'confusion', 'contentment', 'cuteness', 'desire', 'disappointment', 'disgust', 'distress', 'embarassment', 'extasy', 'fear', 'guilt', 'interest', 'neutral', 'pain', 'pride', 'realization', 'relief', 'sadness', 'serenity']
    # ethnicity: ['asian', 'black or african american', 'hispanic or latino', 'prefer not to answer', 'white or caucasian']
    # gender: ['female', 'male', 'non-binary / third gender', 'prefer not to answer']
    # height: ["5' - 5'3", "5'4 - 5'7", "5'8 - 5'11", "6' - 6'3", "< 5'", 'prefer not to answer']
    # native language: ['dari', 'de_de', 'en_gb', 'en_us', 'es', 'prefer not to answer', 'ru', 'ukrainian', 'zh']
    # reading_style: ['emotion', 'fast', 'highpitch', 'loud', 'lowpitch', 'regular', 'slow', 'whisper']
    # speaker_id: p001 - p107 
    # weight: ['100 - 120 lbs', '120 - 140 lbs', '140 - 160 lbs', '160 - 180 lbs', '180 - 200 lbs', '200 - 220 lbs', '220 - 240 lbs', '260 - 280 lbs', '280 - 300 lbs', '340 - 360 lbs', 'prefer not to answer']
    
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


    print("  -> Speaker embedding generated.")

    # Load the prefix audio (silence).
    print("Loading prefix audio...")
    prefix_audio_path = os.path.join("assets", "silence_350ms.wav")
    prefix_audio_codes = model.autoencoder.load_prefix_audio(prefix_audio_path, device=device)
    print(" -> Prefix audio loaded.")
    
    sentence = SpeakerUtils.random_sentence()
    print (f" -> Random sentence: {sentence}")

    # Create and explicitly set the conditioning dictionary.
    cond_dict = make_cond_dict(
        text=sentence, # Hello from Zonos, the state of the art text to speech model.",
        speaker=speaker_embedding,
        language="en_us",
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
        "batch_size": 1,
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

    # Decode the generated codes and save
    model.autoencoder.save_codes("sample_voice_cloning.wav", codes)
    print("Audio saved to sample_voice_cloning.wav")

if __name__ == "__main__":
    main()
