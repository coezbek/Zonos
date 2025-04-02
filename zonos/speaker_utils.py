import os
import json
import re
import torch
import torchaudio
import xxhash
from pathlib import Path

LANGUAGE_MAP = {
    "american english": "en_us",
    "british english":  "en_gb",
    "german":           "de_de",
    "mandarin":         "zh",
    "spanish":          "es",
    "russian":          "ru",
    # Add or adjust as needed...
}

def normalize_language(lang: str) -> str:
    """
    Convert a raw language string to a standard code (e.g. 'en_us').
    If missing, returns the original string or some default code.
    """
    # Lowercase & strip for a robust lookup
    key = lang.lower().strip()
    return LANGUAGE_MAP.get(key, key)  # fallback to the original if not in dict

class SpeakerUtils:
    """
    A utility class for managing speaker embeddings and metadata.
    - Stores embeddings in a directory and metadata in .voices/voices.json.
    - Use get_speaker_embedding with a previously embedded audio file to reload existing embeddings
    - Use load_average to compute the average embedding for a set of speakers based on tags.    
    """
    def __init__(
        self, 
        model, 
        embed_store_dir=".voices"  # default directory
    ):
        self.model = model
        self.embed_store_dir = Path(embed_store_dir)
        self.embed_store_dir.mkdir(parents=True, exist_ok=True)
        self.voices_json_path = self.embed_store_dir / "voices.json"

        self.device = model.device if model else None

    def hash_audio_file(self, filepath: str) -> str:
        """
        Returns a hex digest of the file contents using xxhash xxh3_64.
        NOTE: xxhash is fast but not cryptographically secure.
        """
        hasher = xxhash.xxh3_64()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def embedding_file_path(self, file_hash: str) -> Path:
        return self.embed_store_dir / f"{file_hash[:1]}" / f"{file_hash}.pt"

    def load_embedding_if_exists(self, file_hash: str):
        fpath = self.embedding_file_path(file_hash)
        if fpath.is_file():
            return torch.load(fpath, map_location=self.device)
        return None

    def save_embedding(self, file_hash: str, embedding: torch.Tensor, tags: dict = {}):
        """
        Saves the embedding .pt file and stores metadata in .voices/voices.json
        """
        fpath = self.embedding_file_path(file_hash)
        # Create the directory if it doesn't exist
        fpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding.cpu(), fpath)

        voices_dict = {}
        if self.voices_json_path.is_file():
            with open(self.voices_json_path, "r", encoding="utf-8") as f:
                voices_dict = json.load(f)

        # Store or update the tags for this particular file hash
        voices_dict[file_hash] = tags

        # Save the updated voices.json
        with open(self.voices_json_path, "w", encoding="utf-8") as f:
            json.dump(voices_dict, f, indent=2)

    def get_speaker_embedding(self, audio_file: str, force_recalc=False, tags: dict = {}) -> torch.Tensor:
        """
        Returns an embedding for the given audio file, either by loading from cache or
        calling the model’s `make_speaker_embedding`.
        """
        file_hash = self.hash_audio_file(audio_file)

        if not force_recalc:
            cached_emb = self.load_embedding_if_exists(file_hash)
            if cached_emb is not None:
                return cached_emb

        # Load the audio if no cache or force_recalc is True
        wav, sr = torchaudio.load(audio_file)
        wav = wav.to(self.device)

        with torch.no_grad():
            embedding = self.model.make_speaker_embedding(wav, sr)

        self.save_embedding(file_hash, embedding, tags)
        return embedding

    @staticmethod
    def compute_average(embeddings: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(embeddings, dim=0).mean(dim=0)
    
    @staticmethod
    def random_sentence() -> str:
      """
      Returns a random sentence for testing.
      """
      sentences_list = list(SpeakerUtils.SENTENCES.values())
      random_index = torch.randint(0, len(sentences_list), (1,)).item()
      return sentences_list[random_index]
    

    def scan_speaker_json(self, speaker_stats_json: str):
        """
        - Build a .voices database from Facebook Research EARS dataset.
          - You can download it from https://github.com/facebookresearch/ears_dataset/
          - For use with Zonos, you can download and resample to 16kKz using https://github.com/coezbek/ears_dataset/
        - Reads speaker statistics from speaker_statistics.json file, which is expected to be a dict of path strings to attributes:

          {
            "p001": {
              "gender": "female", ...

        - For each speaker entry (e.g. "p001", "p002"), looks in a directory of the associated names for for .wav files.
        - Wav files must be either named 'emo_(.*)_sentences' for various emotion or 'sentences_01_(.*)' for various reading_styles.
        - Stores all metadata + references to the .pt embedding files in .voices/voices.json.

        Format of voices.json:
          {
            "af5dc...": {
              "age": "36-45",
              "native language": "en_us",
              "emotion": "adoration",
              "transcript": "You're just the sweetest person I know..."
            },
            "6bb12...": { ... },
            ...
          }
        """
        # 1) Load the speaker stats
        with open(speaker_stats_json, "r", encoding="utf-8") as f:
            speaker_data = json.load(f)

        audio_root_dir = Path(speaker_stats_json).parent

        # 2) Load transcripts data (assuming it's in "transcripts.json" next to this script)
        with open(audio_root_dir / "transcripts.json", "r", encoding="utf-8") as f:
            transcripts_data = json.load(f)
            
        # 3) Iterate each speaker in speaker_statistics.json
        for speaker_id, stats in speaker_data.items():
            # Convert "native language" to "native language" (mapped to code if possible)
            if "native language" in stats:
                stats["native language"] = normalize_language(stats["native language"])

            # 4) For each transcript name, see if a .wav file exists at audio_root_dir/speaker_id
            for audio_name, sentence in transcripts_data.items():
                my_stats = stats.copy()

                print(f"Processing {speaker_id}/{audio_name}...")

                # Possibly parse "emo_(.*)_sentences" for emotion
                if x := re.search(r"emo_(.*)_sentences", audio_name):
                    emotion = x.group(1)
                    my_stats["emotion"] = emotion
                    my_stats["reading_style"] = "emotion"
                
                # Possibly parse for reading style like "sentences_01_fast", etc.
                if x := re.search(r"(sentences|rainbow)_\d\d_(.*)", audio_name):
                    reading_style = x.group(2)
                    my_stats["reading_style"] = reading_style
                
                path = os.path.join(audio_root_dir, speaker_id, audio_name + ".wav")

                # Save the transcript itself
                my_stats["transcript"] = sentence
                my_stats["original_path"] = path
                my_stats["speaker_id"] = speaker_id

                # Build the path to the WAV file

                if not os.path.isfile(path):
                    print(f"Warning: File {path} not found. Skipping.")
                    continue
                
                # Force recalculation each time for demonstration.  If you prefer to keep caches,
                # set force_recalc=False or provide an option.
                self.get_speaker_embedding(path, force_recalc=True, tags=my_stats)

        print(f"Scan complete. Wrote metadata to {self.voices_json_path}")

    def print_tags(self):
        """
        Prints all unique tags from the voices.json file.
        """
        if not self.voices_json_path.is_file():
            raise FileNotFoundError(f"No voices.json found at {self.voices_json_path}")

        with open(self.voices_json_path, "r", encoding="utf-8") as f:
            voices_dict = json.load(f)

        all_keys = set()
        all_tags = dict()

        for speaker_tags in voices_dict.values():
            all_keys.update(speaker_tags.keys())

            for k, v in speaker_tags.items():
                if k not in all_tags:
                    all_tags[k] = set()
                all_tags[k].add(v)

        all_keys = all_keys - {"original_path", "transcript"}

        print("Unique tags in voices.json:")
        for key in sorted(all_keys):
            print(f" - {key}: { sorted(all_tags[key]) }")

    def load_average(self, tags: dict) -> torch.Tensor:
        """
        - Reads .voices/voices.json
        - Finds all speakers whose "tags" match the given `tags` dict (exact match for k/v)
        - Loads all embedding .pt files from those matched speakers
        - Returns the average embedding (torch.Tensor)

        Example usage:
          avg_female_3645 = speaker_utils.load_average({"gender": "female", "age": "36-45"})
          speaker_p001 = speaker_utils.load_average({"speaker_id": "p001", "reading_style": "regular"})
          speaker_p001_whisper = speaker_utils.load_average({"speaker_id": "p001", "reading_style": "whisper"})

        """
        if not self.voices_json_path.is_file():
            raise FileNotFoundError(f"No voices.json found at {self.voices_json_path}. Need to download EARS dataset and scan() first.")

        with open(self.voices_json_path, "r", encoding="utf-8") as f:
            voices_dict = json.load(f)

        matched_embeddings = []
        for hash_id, speaker_tags in voices_dict.items():
            # If all requested tags match exactly, we load that embedding
            if all(speaker_tags.get(k) == v for k, v in tags.items()):
                emb = self.load_embedding_if_exists(hash_id)
                if emb is not None:
                    matched_embeddings.append(emb)
                else: 
                    print(f"Warning: Embedding file for {hash_id} not found. Skipping.")
        
        if not matched_embeddings:
            raise ValueError(f"No matching embeddings found for tags: {tags}")
        
        print(f"Found {len(matched_embeddings)} matching embeddings for tags: {tags}")

        return self.compute_average(matched_embeddings)
    
    SENTENCES = {
          "emo_adoration_sentences": "You're just the sweetest person I know and I am so happy to call you my friend. I had the best time with you, I just adore you. I love this gift, thank you!",
          "emo_amazement_sentences": "I just love how you can play guitar. You're so impressive. I admire your abilities so much.",
          "emo_amusement_sentences": "The sound that baby just made was quite amusing. I liked that stand up comic, I found her pretty funny. What a fun little show to watch!",
          "emo_anger_sentences": "I'm so mad right now I could punch a hole in the wall. I can't believe he said that, he's such a jerk! There's a stop sign there and parents are just letting their kids run around!",
          "emo_confusion_sentences": "Huh, what is going on over here? What is this? Where are we going?",
          "emo_contentment_sentences": "I really enjoyed dinner tonight, it was quite nice. Everything is working out just fine. I'm good either way.",
          "emo_cuteness_sentences": "Look at that cute little kitty cat! Oh my goodness, she's so cute! That's the cutest thing I've ever seen!",
          "emo_desire_sentences": "Mmm that chocolate fudge lava cake looks devine. I want that car so badly. I can't wait to see you again.",
          "emo_disappointment_sentences": "I'm so disappointed in myself. I wish I had worked harder. I had such higher expectations for you. I really was hoping you were better than this.",
          "emo_disgust_sentences": "I have never seen anything grosser than this in my entire life. This is the worst dinner I've ever had. Yuck, I can't even look at that.",
          "emo_distress_sentences": "Oh god, I am not sure if we are going to make this flight on time. This is all too stressful to handle right now. I don't know where anything is and I'm running late.",
          "emo_embarassment_sentences": "I don't know what happend, I followed the recipe perfectly but the cake just deflated. I'm so embarrassed. I hope no one saw that, I'd be mortified if they did.",
          "emo_extasy_sentences": "This is the most exciting thing I've ever seen in my life! I can't believe I got to see that. I'm so excited, I've never been there before.",
          "emo_fear_sentences": "Did you hear that sound? I'm afraid someone or something is outside. Oh my gosh, what is that? What do you think is going to happen if we don't run?",
          "emo_guilt_sentences": "I'm sorry I did that to you. I really didn't mean to hurt you. I feel horrible that happened to you.",
          "emo_interest_sentences": "Hmm, I wonder what that cookie tastes like. Oh, what is that over there? So what exactly is it that you do?",
          "emo_neutral_sentences": "That wall in the living room is white. There is one more piece of bread in the pantry. The store closes at 8pm tonight.",
          "emo_pain_sentences": "Oh, this headache is the worst one I've ever had! My foot hurts so badly right now! I'm in terrible pain from that medication.",
          "emo_pride_sentences": "That was all me, I'm the one who found the project, created the company and made it succeed. I have worked hard to get here and I deserve it. I'm really proud of how well you did.",
          "emo_realization_sentences": "Wow, I never know that the body was made up of 75% water. Did you know that a flamingo is actually white but turns pink because it eats too many shrimp? Apparently dolphins sleep with one eye open.",
          "emo_relief_sentences": "I'm so relieved my taxes are done. That was so stressful. I'm so relieved that is over with. Thank goodness that's all done.",
          "emo_sadness_sentences": "I am so upset by the state of the world. I hope it gets better soon. I really miss her, life isn't the same without her. I'm sorry for your loss.",
          "emo_serenity_sentences": "This has been the most peaceful day of my life. I am very calm right now. I'm going to relax and take a nap here on the beach.",
          "rainbow_01_fast": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
          "rainbow_01_highpitch": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
          "rainbow_01_loud": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
          "rainbow_01_lowpitch": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
          "rainbow_01_regular": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
          "rainbow_01_slow": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
          "rainbow_01_whisper": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
          "rainbow_02_fast": "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
          "rainbow_02_highpitch": "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
          "rainbow_02_loud": "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
          "rainbow_02_lowpitch": "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
          "rainbow_02_regular": "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
          "rainbow_02_slow": "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
          "rainbow_02_whisper": "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
          "rainbow_03_fast": "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
          "rainbow_03_highpitch": "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
          "rainbow_03_loud": "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
          "rainbow_03_lowpitch": "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
          "rainbow_03_regular": "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
          "rainbow_03_slow": "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
          "rainbow_03_whisper": "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
          "rainbow_04_fast": "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
          "rainbow_04_highpitch": "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
          "rainbow_04_loud": "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
          "rainbow_04_lowpitch": "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
          "rainbow_04_regular": "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
          "rainbow_04_slow": "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
          "rainbow_04_whisper": "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
          "rainbow_05_fast": "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
          "rainbow_05_highpitch": "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
          "rainbow_05_loud": "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
          "rainbow_05_lowpitch": "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
          "rainbow_05_regular": "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
          "rainbow_05_slow": "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
          "rainbow_05_whisper": "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
          "rainbow_06_fast": "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
          "rainbow_06_highpitch": "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
          "rainbow_06_loud": "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
          "rainbow_06_lowpitch": "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
          "rainbow_06_regular": "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
          "rainbow_06_slow": "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
          "rainbow_06_whisper": "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
          "rainbow_07_fast": "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
          "rainbow_07_highpitch": "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
          "rainbow_07_loud": "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
          "rainbow_07_lowpitch": "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
          "rainbow_07_regular": "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
          "rainbow_07_slow": "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
          "rainbow_07_whisper": "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
          "rainbow_08_fast": "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
          "rainbow_08_highpitch": "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
          "rainbow_08_loud": "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
          "rainbow_08_lowpitch": "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
          "rainbow_08_regular": "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
          "rainbow_08_slow": "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
          "rainbow_08_whisper": "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
          "sentences_01_fast": "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
          "sentences_01_highpitch": "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
          "sentences_01_loud": "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
          "sentences_01_lowpitch": "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
          "sentences_01_regular": "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
          "sentences_01_slow": "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
          "sentences_01_whisper": "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
          "sentences_02_fast": "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
          "sentences_02_highpitch": "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
          "sentences_02_loud": "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
          "sentences_02_lowpitch": "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
          "sentences_02_regular": "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
          "sentences_02_slow": "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
          "sentences_02_whisper": "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
          "sentences_03_whisper": "Had it been but one, it had been easy. We have boxed the compass among us. I shall rush out and prevent it. All that is mean slander. The doctor seemed tired and in a hurry.",
          "sentences_04_whisper": "I only heard it last night. We had now got into the month of March. But go thy ways; I had forgot. Conceited fellow with his waxed up moustache! Anne's unhappiness continued for a week.",
          "sentences_05_loud": "In fact, the count's face brightened. For God's sake, talk to her. In what an amiable light does this place him! Take me out of my way. I heard many things in hell.",
          "sentences_06_loud": "Yes; but we do not invite people of fashion. You see what he writes. Silent with awe and pity I went to her bedside. Happy to say, I never knew him. Birthdays are of no importance to a rational being.",
          "sentences_07_slow": "But it may all be put in two words. Clear up the room, the sick man said with effort. He was still in sight. He delayed; he seemed almost afraid of something. Then they carried me in.",
          "sentences_08_slow": "But I have never been presented. But we were only in fun! Now, look at that third name. And serve them both right, too. Good glass of burgundy take away that.",
          "sentences_09_fast": "And it seemed to her that God heard her prayer. My word, I admire you. I also have a pious visit to pay. She has promised to come on the twentieth. I want to tell you something.",
          "sentences_10_fast": "Oh, sir, it will break bones. I am very glad to see you. This question absorbed all his mental powers. Before going away forever, I'll tell him all. I told you it was mother.",
          "sentences_11_highpitch": "You're all in good spirits. They might retreat and leave the pickets. But I like sentimental people. Our potato crop is very good this year. Why is the chestnut on the right?",
          "sentences_12_highpitch": "His room was on the first floor. I have had a pattern in my hand. The knocking still continued and grew louder. May my sorrows ever shun the light. How must I arrange it, then?",
          "sentences_13_lowpitch": "Just read it out to me. I shall take your advice in every particular. What mortal imagination could conceive it? The gate was again hidden by smoke. After a while I left him.",
          "sentences_14_lowpitch": "There was a catch in her breath. They told me, but I didn't understand. What a cabin it is. A cry of joy broke from his lips. He had obviously prepared the sentence beforehand.",
          "sentences_15_regular": "They were all sitting in her room. So that's how it stands. He did not know why he embraced it. Why don't you speak, cousin? I didn't tell a tale.",
          "sentences_16_regular": "My head aches dreadfully now. Not to say every word. I have only found out. He is trying to discover something. I have done my duty.",
          "sentences_17_regular": "I always had a value for him. He is a deceiver and a villain. But those tears were pleasant to them both. She conquered her fears, and spoke. Oh, he couldn't overhear me at the door.",
          "sentences_18_regular": "How could I have said it more directly? She remembered her oath. My kingdom for a drink! Have they caught the little girl and the boy? Then she gave him the dry bread.",
          "sentences_19_regular": "Your sister is given to government. Water was being sprinkled on his face. The clumsy things are dear. He jumped up and sat on the sofa. How do you know her?",
          "sentences_20_regular": "I never could guess a riddle in my life. The expression of her face was cold. Besides, what on earth could happen to you? Allow me to give you a piece of advice. This must be stopped at once.",
          "sentences_21_regular": "The lawyer was right about that. You are fond of fighting. Every word is so deep. So you were never in London before? Death is now, perhaps, striking a fourth blow.",
          "sentences_22_regular": "It seemed that sleep and night had resumed their empire. The snowstorm was still raging. But we'll talk later on. Take the baby, Mum, and give me your book. The doctor gave him his hand.",
          "sentences_23_regular": "It is, nevertheless, conclusive to my mind. Give this to the countess. It is only a question of a few hours. No, we don't keep a cat. The cool evening air refreshed him.",
          "sentences_24_regular": "You can well enjoy the evening now. We'll make up for it now. The weakness of a murderer. But they wouldn't leave me alone. The telegram was from his wife.",
        }

################################################################################
# Optional: A main() function to run from the command line
################################################################################
def main():
    parser = argparse.ArgumentParser(description="Scan a dataset and build speaker embeddings.")
    parser.add_argument(
        "--speaker_stats",
        help="Path to speaker_statistics.json"
    )
    parser.add_argument(
        "--embed_store_dir",
        default=".voices",
        help="Directory to store embeddings and voices.json"
    )
    parser.add_argument(
        "--print_tags",
        action="store_true",
        help="Print all unique tags from voices.json"
    )

    args = parser.parse_args()

    if args.print_tags:
        # 1) Load the Zonos model
        # NOTE: This will load the model to the fastest available device (GPU or CPU)
        # from zonos.utils import set_device
        # set_device("memory") # Set device to fastest available GPU or CPU
        from zonos.utils import DEFAULT_DEVICE as device

        spk_utils = SpeakerUtils(
            model=None,
            embed_store_dir=args.embed_store_dir
        )
        spk_utils.print_tags()
        return
    
    if args.speaker_stats is None:
        raise ValueError("Please provide --speaker_stats path to speaker_statistics.json")
    
    # from zonos.utils import set_device
    # set_device("memory") # Set device to fastest available GPU or CPU
    from zonos.utils import DEFAULT_DEVICE as device
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device).eval().requires_grad_(False)

    # 2) Create SpeakerUtils
    spk_utils = SpeakerUtils(
        model=model,
        embed_store_dir=args.embed_store_dir,
    )

    # 3) Scan the dataset
    spk_utils.scan_speaker_json(args.speaker_stats)

    # Example: you could do a post-run test of load_average if you want:
    #   try:
    #       avg_female_3645 = spk_utils.load_average({"gender": "female", "age": "36-45"})
    #       print("Average embedding shape:", avg_female_3645.shape)
    #   except ValueError as e:
    #       print(e)

if __name__ == "__main__":
    import argparse  # For command line parsing
    from zonos.model import Zonos
    main()
