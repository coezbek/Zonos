import os
import json
import re
import torch
import torchaudio
import xxhash  # pip install xxhash
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
    def __init__(
        self, 
        model, 
        embed_store_dir=".voices",  # default directory
        device="cuda"
    ):
        self.model = model.eval().requires_grad_(False).to(device)
        self.embed_store_dir = Path(embed_store_dir)
        self.embed_store_dir.mkdir(parents=True, exist_ok=True)
        self.voices_json_path = self.embed_store_dir / "voices.json"

        self.device = device

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
        return self.embed_store_dir / f"{file_hash}.pt"

    def load_embedding_if_exists(self, file_hash: str):
        fpath = self.embedding_file_path(file_hash)
        if fpath.is_file():
            return torch.load(fpath, map_location=self.device)
        return None

    def save_embedding(self, file_hash: str, embedding: torch.Tensor, tags: dict = {}):
        fpath = self.embedding_file_path(file_hash)
        torch.save(embedding.cpu(), fpath)

        voices_dict = {}
        if self.voices_json_path.is_file():
            with open(self.voices_json_path, "r", encoding="utf-8") as f:
                voices_dict = json.load(f)

        voices_dict[file_hash] = tags

        # Save the updated voices.json
        with open(self.voices_json_path, "w", encoding="utf-8") as f:
            json.dump(voices_dict, f, indent=2)

    def get_speaker_embedding(self, audio_file: str, force_recalc=False, tags: dict = {}) -> torch.Tensor:
        file_hash = self.hash_audio_file(audio_file)

        if not force_recalc:
            cached_emb = self.load_embedding_if_exists(file_hash)
            if cached_emb is not None:
                return cached_emb

        wav, sr = torchaudio.load(audio_file)
        wav = wav.to(self.device)

        with torch.no_grad():
            embedding = self.model.make_speaker_embedding(wav, sr)

        self.save_embedding(file_hash, embedding, tags)
        return embedding

    @staticmethod
    def compute_average(embeddings: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(embeddings, dim=0).mean(dim=0)

    def scan(self, speaker_stats_json: str, audio_root_dir: str):
        """
        - Reads speaker statistics from `speaker_stats_json` (e.g. speaker_statistics.json).
        - For each speaker entry (e.g. "p001", "p002"), looks in `audio_root_dir/pXXX` for .wav files.
        - Generates embeddings via self.get_speaker_embedding().
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

        with open("transcripts.json", "r", encoding="utf-8") as f:
            transcripts_data = json.load(f)
            
        # 3) Iterate each speaker in speaker_statistics.json
        for speaker_id, stats in speaker_data.items():
            # Convert "native language" to "native_language"
            # Map it to a standard code if possible

            if "native language" in stats:
                stats["native language"] = normalize_language(stats["native language"])

            for audio_name, sentence in transcripts_data.items():

                my_stats = stats.copy()

                # Ruby regexp would be: if audio_name =~ /emo_(.*?)_sentences/
                # Python equivalent:
                if x := re.search(r"emo_(.*)_sentences", audio_name):
                    # Extract the emotion
                    emotion = x.group(1)
                    my_stats["emotion"] = emotion
                    my_stats["reading_style"] = "emotion"
                
                if x := re.search(r"(sentences|rainbow)_\d\d_(.*)", audio_name):
                  # Extract the reading style
                  reading_style = x.group(2)
                  my_stats["reading_style"] = reading_style
                
                my_stats["transcript"] = sentence

                path = os.path.join(audio_root_dir, speaker_id, audio_name + ".wav")
                # Check if the audio file exists
                if not os.path.isfile(path):
                    print(f"Warning: File {path} not found. Skipping.")
                    continue
                
                self.get_speaker_embedding(path, force_recalc=True, tags=my_stats)

        print(f"Scan complete. Wrote metadata to {self.voices_json_path}")

    ############################################################################
    # 2) LOAD_AVERAGE: Filter by tags, load those embeddings, return average
    ############################################################################
    def load_average(self, tags: dict) -> torch.Tensor:
        """
        - Reads .voices/voices.json
        - Finds all speakers whose "tags" match the given `tags` dict (exact match for k/v)
        - Loads all embedding .pt files from those matched speakers
        - Returns the average embedding (torch.Tensor)

        Example usage:
          avg_female_3645 = speaker_utils.load_average({"gender": "female", "age": "36-45"})
        """

        if not self.voices_json_path.is_file():
            raise FileNotFoundError(f"No voices.json found at {self.voices_json_path}")

        with open(self.voices_json_path, "r", encoding="utf-8") as f:
            voices_dict = json.load(f)

        matched_embeddings = []
        for hash_id, speaker_tags in voices_dict.items():

            # Check if the user-supplied tags are all present + matching
            # e.g. if tags = {"gender": "female"} then speaker_tags["gender"] should be "female"
            if all(speaker_tags.get(k) == v for k, v in tags.items()):

                emb = self.load_embedding_if_exists(hash_id)
                if emb is not None:
                    matched_embeddings.append(emb)
                else: 
                    print(f"Warning: Embedding file for {hash_id} not found. Skipping.")
        
        if not matched_embeddings:
            raise ValueError(f"No matching embeddings found for tags: {tags}")

        # Average them
        return self.compute_average(matched_embeddings)
