import os
import json
import re
import torch
import random
import torchaudio
import xxhash
from datasets import load_dataset
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
        # Default dir is ../.voices relative to this file
        embed_store_dir=Path(__file__).parent.parent / ".voices",
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

    def is_audio_hash(self, path_or_string: str) -> bool:
        pattern = r'^[0-9a-fA-F]{16}(\.pt)?$'
        return re.fullmatch(pattern, path_or_string) is not None

    def get_speaker_embedding(self, audio_file: str, force_recalc=False, tags: dict = {}) -> torch.Tensor:
        """
        Returns an embedding for the given audio file, either by loading from cache or
        calling the model's `make_speaker_embedding` for the given filepath.

        If the audio_file is a directory, it computes the average embedding for all files in that directory.

        If the audio_file is a hash, it loads the embedding from the cache.
        """

        if self.is_audio_hash(audio_file):
            file_hash = audio_file
        else:
            # Check if directory:
            if os.path.isdir(audio_file):
                return self.compute_average(
                    [self.get_speaker_embedding(os.path.join(audio_file, f), force_recalc, tags) for f in os.listdir(audio_file)]
                )
            else:
                file_hash = self.hash_audio_file(audio_file)

        # Cut-off .pt in case a hash is passed
        if file_hash.endswith(".pt"):
            file_hash = file_hash[:-3]

        if not force_recalc:
            cached_emb = self.load_embedding_if_exists(file_hash)
            if cached_emb is not None:
                return cached_emb

        # Load the audio if no cache or force_recalc is True
        wav, sr = torchaudio.load(audio_file)
        wav = wav.to(self.device)

        # If necessary convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Pad audio end with 100 ms of silence at the end
        silence_duration = 0.1
        silence = torch.zeros(int(silence_duration * sr), device=self.device)
        wav = torch.cat((wav, silence.unsqueeze(0)), dim=1)

        with torch.no_grad():
            embedding = self.model.make_speaker_embedding(wav, sr)

        self.save_embedding(file_hash, embedding, tags)
        return embedding

    @staticmethod
    def compute_average(embeddings: list[torch.Tensor]) -> torch.Tensor:
        if len(embeddings) == 1:
          return embeddings[0]

        return torch.stack(embeddings, dim=0).mean(dim=0)
    
    @staticmethod
    def random_sentence(lang='en') -> str:
      """
      Returns a random sentence for testing.
      """
      # Keep only language code, drop country code (e.g. 'en_us' -> 'en')
      lang = lang.split('_')[0]

      if lang == 'en' or lang == 'de':  
        sentences_list = list(SpeakerUtils.SENTENCES[lang].values())
        random_index = torch.randint(0, len(sentences_list), (1,)).item()
        return sentences_list[random_index]
      else:
        # Load the dataset for the specified language
        dataset = load_dataset('agentlans/high-quality-multilingual-sentences', data_files=[f'{lang}.jsonl.zst'])
        # DatasetDict({
        #   train: Dataset({
        #       features: ['text'],
        #       num_rows: 44410
        #   })
        # })
        
        # Extract the list of sentences
        sentences = dataset['train']['text']
        
        # Select and return a random sentence
        return random.choice(sentences)      

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
            raise ValueError(f"No matching embeddings found for tags: {tags} in {len(voices_dict.items())} embeddings.")
        
        
        print(f"Found {len(matched_embeddings)} matching embeddings for tags: {tags}")

        return self.compute_average(matched_embeddings)
    
    SENTENCES = {
      "en": {
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
          "emo_neutral_sentences": "That wall in the living room is white. There is one more piece of bread in the pantry. The store closes at eight p m tonight.",
          "emo_pain_sentences": "Oh, this headache is the worst one I've ever had! My foot hurts so badly right now! I'm in terrible pain from that medication.",
          "emo_pride_sentences": "That was all me, I'm the one who found the project, created the company and made it succeed. I have worked hard to get here and I deserve it. I'm really proud of how well you did.",
          "emo_realization_sentences": "Wow, I never know that the body was made up of seventy five percent water. Did you know that a flamingo is actually white but turns pink because it eats too many shrimp? Apparently dolphins sleep with one eye open.",
          "emo_relief_sentences": "I'm so relieved my taxes are done. That was so stressful. I'm so relieved that is over with. Thank goodness that's all done.",
          "emo_sadness_sentences": "I am so upset by the state of the world. I hope it gets better soon. I really miss her, life isn't the same without her. I'm sorry for your loss.",
          "emo_serenity_sentences": "This has been the most peaceful day of my life. I am very calm right now. I'm going to relax and take a nap here on the beach.",
          "rainbow_01": "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
          "rainbow_02": "These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end.",
          "rainbow_03": "People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow. Throughout the centuries people have explained the rainbow in various ways.",
          "rainbow_04": "Some have accepted it as a miracle without physical explanation. To the Hebrews it was a token that there would be no more universal floods. The Greeks used to imagine that it was a sign from the gods to foretell war or heavy rain.",
          "rainbow_05": "The Norsemen considered the rainbow as a bridge over which the gods passed from earth to their home in the sky. Others have tried to explain the phenomenon physically. Aristotle thought that the rainbow was caused by reflection of the sun's rays by the rain.",
          "rainbow_06": "Since then physicists have found that it is not reflection, but refraction by the raindrops which causes the rainbows. Many complicated ideas about the rainbow have been formed.",
          "rainbow_07": "The difference in the rainbow depends considerably upon the size of the drops, and the width of the colored band increases as the size of the drops increases. The actual primary rainbow observed is said to be the effect of super-imposition of a number of bows.",
          "rainbow_08": "If the red of the second bow falls upon the green of the first, the result is to give a bow with an abnormally wide yellow band, since red and green light when mixed form yellow. This is a very common type of bow, one showing mainly red and yellow, with little or no green or blue.",
          "sentences_01": "I will not stay here. God, we simply must dress the character. Stay, stay, I will go myself. May one ask what it is for. He rushed to the window and opened the movable pane.",
          "sentences_02": "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
          "sentences_02": "It might happen, he added with an involuntary smile. It is sold, sir, was again his laconic reply. And you must have some water, my dear fellow. What is that flying about? Who wants a dead cert for the Gold cup?",
          "sentences_03": "Had it been but one, it had been easy. We have boxed the compass among us. I shall rush out and prevent it. All that is mean slander. The doctor seemed tired and in a hurry.",
          "sentences_04": "I only heard it last night. We had now got into the month of March. But go thy ways; I had forgot. Conceited fellow with his waxed up moustache! Anne's unhappiness continued for a week.",
          "sentences_05": "In fact, the count's face brightened. For God's sake, talk to her. In what an amiable light does this place him! Take me out of my way. I heard many things in hell.",
          "sentences_06": "Yes; but we do not invite people of fashion. You see what he writes. Silent with awe and pity I went to her bedside. Happy to say, I never knew him. Birthdays are of no importance to a rational being.",
          "sentences_07": "But it may all be put in two words. Clear up the room, the sick man said with effort. He was still in sight. He delayed; he seemed almost afraid of something. Then they carried me in.",
          "sentences_08": "But I have never been presented. But we were only in fun! Now, look at that third name. And serve them both right, too. Good glass of burgundy take away that.",
          "sentences_09": "And it seemed to her that God heard her prayer. My word, I admire you. I also have a pious visit to pay. She has promised to come on the twentieth. I want to tell you something.",
          "sentences_10": "Oh, sir, it will break bones. I am very glad to see you. This question absorbed all his mental powers. Before going away forever, I'll tell him all. I told you it was mother.",
          "sentences_11": "You're all in good spirits. They might retreat and leave the pickets. But I like sentimental people. Our potato crop is very good this year. Why is the chestnut on the right?",
          "sentences_12": "His room was on the first floor. I have had a pattern in my hand. The knocking still continued and grew louder. May my sorrows ever shun the light. How must I arrange it, then?",
          "sentences_13": "Just read it out to me. I shall take your advice in every particular. What mortal imagination could conceive it? The gate was again hidden by smoke. After a while I left him.",
          "sentences_14": "There was a catch in her breath. They told me, but I didn't understand. What a cabin it is. A cry of joy broke from his lips. He had obviously prepared the sentence beforehand.",
          "sentences_15": "They were all sitting in her room. So that's how it stands. He did not know why he embraced it. Why don't you speak, cousin? I didn't tell a tale.",
          "sentences_16": "My head aches dreadfully now. Not to say every word. I have only found out. He is trying to discover something. I have done my duty.",
          "sentences_17": "I always had a value for him. He is a deceiver and a villain. But those tears were pleasant to them both. She conquered her fears, and spoke. Oh, he couldn't overhear me at the door.",
          "sentences_18": "How could I have said it more directly? She remembered her oath. My kingdom for a drink! Have they caught the little girl and the boy? Then she gave him the dry bread.",
          "sentences_19": "Your sister is given to government. Water was being sprinkled on his face. The clumsy things are dear. He jumped up and sat on the sofa. How do you know her?",
          "sentences_20": "I never could guess a riddle in my life. The expression of her face was cold. Besides, what on earth could happen to you? Allow me to give you a piece of advice. This must be stopped at once.",
          "sentences_21": "The lawyer was right about that. You are fond of fighting. Every word is so deep. So you were never in London before? Death is now, perhaps, striking a fourth blow.",
          "sentences_22": "It seemed that sleep and night had resumed their empire. The snowstorm was still raging. But we'll talk later on. Take the baby, Mum, and give me your book. The doctor gave him his hand.",
          "sentences_23": "It is, nevertheless, conclusive to my mind. Give this to the countess. It is only a question of a few hours. No, we don't keep a cat. The cool evening air refreshed him.",
          "sentences_24": "You can well enjoy the evening now. We'll make up for it now. The weakness of a murderer. But they wouldn't leave me alone. The telegram was from his wife.",
        },
      "de": {
          "emo_adoration_sentences": "Du bist einfach die süßeste Person, die ich kenne, und ich freue mich so, dich meine Freundin nennen zu dürfen. Ich hatte die beste Zeit mit dir, ich verehre dich einfach. Ich liebe dieses Geschenk, danke!",
          "emo_amazement_sentences": "Ich liebe es einfach, wie du Gitarre spielen kannst. Du bist so beeindruckend. Ich bewundere deine Fähigkeiten sehr.",
          "emo_amusement_sentences": "Das Geräusch, das das Baby gerade gemacht hat, war ziemlich amüsant. Ich mochte diese Stand-up-Comedian, ich fand sie ziemlich lustig. Was für eine lustige kleine Show!",
          "emo_anger_sentences": "Ich bin gerade so wütend, dass ich ein Loch in die Wand schlagen könnte. Ich kann nicht fassen, dass er das gesagt hat, er ist so ein Idiot! Da ist ein Stoppschild und Eltern lassen ihre Kinder einfach herumlaufen!",
          "emo_confusion_sentences": "Hä, was geht hier vor sich? Was ist das? Wohin gehen wir?",
          "emo_contentment_sentences": "Ich habe das Abendessen heute Abend wirklich genossen, es war ziemlich nett. Alles läuft gerade gut. Ich bin in Ordnung, wie auch immer.",
          "emo_cuteness_sentences": "Schau dir diese süße kleine Katze an! Oh mein Gott, sie ist so süß! Das ist das süßeste, was ich je gesehen habe!",
          "emo_disappointment_sentences": "Ich bin so enttäuscht von mir selbst. Ich wünschte, ich hätte härter gearbeitet. Ich hatte viel höhere Erwartungen an dich. Ich hatte wirklich gehofft, dass du besser bist als das.",
          "emo_disgust_sentences": "Ich habe noch nie etwas Ekligeres gesehen als das in meinem ganzen Leben. Das ist das schlimmste Abendessen, das ich je hatte. Iiiih, ich kann das nicht einmal anschauen.",
          "emo_distress_sentences": "Oh Gott, ich bin mir nicht sicher, ob wir diesen Flug rechtzeitig erreichen werden. Das ist gerade alles zu stressig, um es zu bewältigen. Ich weiß nicht, wo etwas ist und ich komme zu spät.",
          "emo_embarassment_sentences": "Ich weiß nicht, was passiert ist, ich habe das Rezept perfekt befolgt, aber der Kuchen ist einfach zusammengefallen. Es ist mir so peinlich. Ich hoffe, niemand hat das gesehen, ich wäre entsetzt, wenn sie es getan hätten.",
          "emo_extasy_sentences": "Das ist das aufregendste, was ich je in meinem Leben gesehen habe! Ich kann es nicht fassen, dass ich das gesehen habe. Ich bin so aufgeregt, ich war noch nie dort.",
          "emo_fear_sentences": "Hast du dieses Geräusch gehört? Ich habe Angst, dass jemand oder etwas draußen ist. Oh mein Gott, was ist das? Was glaubst du, wird passieren, wenn wir nicht weglaufen?",
          "emo_guilt_sentences": "Es tut mir leid, dass ich dir das angetan habe. Ich wollte dir wirklich nicht weh tun. Ich fühle mich furchtbar, dass das dir passiert ist.",
          "emo_interest_sentences": "Hmm, ich frage mich, wie dieser Keks schmeckt. Oh, was ist das da drüben? Was genau machst du eigentlich?",
          "emo_neutral_sentences": "Die Wand im Wohnzimmer ist weiß. Es gibt noch ein Stück Brot in der Speisekammer. Der Laden schließt heute um zwanzig Uhr.",
          "emo_pain_sentences": "Oh, dieser Kopfschmerz ist der schlimmste, den ich je hatte! Mein Fuß tut so weh gerade! Ich habe so schlimme Schmerzen von diesem Medikament.",
          "emo_pride_sentences": "Das war alles ich, ich bin derjenige, der das Projekt gefunden hat, das Unternehmen gegründet hat und es erfolgreich gemacht hat. Ich habe hart gearbeitet, um hierher zu kommen, und ich verdiene es. Ich bin wirklich stolz auf das, was du erreicht hast.",
          "emo_realization_sentences": "Wow, ich wusste nicht, dass der Körper zu fünf und siebzig Prozent aus Wasser besteht. Wusstest du, dass ein Flamingo eigentlich weiß ist, aber rosa wird, weil er zu viele Garnelen isst? Angeblich schlafen Delfine mit einem Auge offen.",
          "emo_relief_sentences": "Ich bin so erleichtert, dass meine Steuern erledigt sind. Das war so stressig. Ich bin so erleichtert, dass das vorbei ist. Gott sei Dank, das ist alles erledigt.",
          "emo_sadness_sentences": "Ich bin so traurig über den Zustand der Welt. Ich hoffe, es wird bald besser. Ich vermisse sie wirklich, das Leben ist nicht mehr dasselbe ohne sie. Es tut mir leid für deinen Verlust.",
          "emo_serenity_sentences": "Dies war der friedlichste Tag meines Lebens. Ich bin gerade sehr ruhig. Ich werde mich entspannen und hier am Strand ein Nickerchen machen.",
          "rainbow_01": "Wenn das Sonnenlicht auf Regentropfen in der Luft trifft, wirken sie wie ein Prisma und bilden einen Regenbogen. Der Regenbogen ist eine Aufspaltung des weißen Lichts in viele wunderschöne Farben.",
          "rainbow_02": "Diese nehmen die Form eines langen, runden Bogens an, dessen Verlauf hoch oben liegt und dessen beide Enden scheinbar jenseits des Horizonts liegen. Der Legende nach befindet sich an einem Ende ein kochender Goldtopf.",
          "rainbow_03": "Die Menschen schauen danach, aber niemand findet ihn jemals. Wenn jemand etwas sucht, das außerhalb seiner Reichweite liegt, sagen seine Freunde, er suche den Goldtopf am Ende des Regenbogens. Im Laufe der Jahrhunderte haben Menschen den Regenbogen auf unterschiedliche Weise erklärt.",
          "rainbow_04": "Einige akzeptierten ihn als Wunder ohne physikalische Erklärung. Für die Hebräer war er ein Zeichen dafür, dass es keine weiteren Sintfluten mehr geben würde. Die Griechen glaubten, es sei ein Zeichen der Götter, das Krieg oder starken Regen ankündigte.",
          "rainbow_05": "Die Nordmänner betrachteten den Regenbogen als Brücke, über die die Götter von der Erde in ihr Zuhause am Himmel gelangten. Andere versuchten, das Phänomen physikalisch zu erklären. Aristoteles dachte, der Regenbogen entstehe durch die Reflexion der Sonnenstrahlen im Regen.",
          "rainbow_06": "Seitdem haben Physiker festgestellt, dass es nicht die Reflexion, sondern die Brechung in den Regentropfen ist, die Regenbögen erzeugt. Viele komplizierte Vorstellungen über den Regenbogen wurden entwickelt.",
          "rainbow_07": "Die Unterschiede beim Regenbogen hängen maßgeblich von der Größe der Tropfen ab, und die Breite des farbigen Bandes nimmt mit der Größe der Tropfen zu. Der tatsächlich beobachtete primäre Regenbogen ist angeblich das Ergebnis einer Überlagerung mehrerer Bögen.",
          "rainbow_08": "Wenn das Rot des zweiten Bogens auf das Grün des ersten fällt, entsteht ein Bogen mit einem ungewöhnlich breiten gelben Band, da rotes und grünes Licht zusammen Gelb ergeben. Dies ist ein sehr häufiger Regenbogentyp, der hauptsächlich Rot und Gelb zeigt und wenig oder gar kein Grün oder Blau enthält.",
          "sentences_01": "Ich werde nicht hierbleiben. Gott, wir müssen den Charakter unbedingt einkleiden. Bleib, bleib, ich gehe selbst. Darf man fragen, wozu das dient? Er eilte zum Fenster und öffnete die bewegliche Scheibe.",
          "sentences_02": "Es könnte passieren, fügte er mit einem unwillkürlichen Lächeln hinzu. Es ist verkauft, Herr, war wiederum seine lakonische Antwort. Und du musst etwas Wasser haben, mein lieber Freund. Was fliegt denn da herum? Wer möchte einen sicheren Tipp für den Gold Cup?",
          "sentences_03": "Wäre es nur einer gewesen, wäre es leicht gewesen. Wir haben gemeinsam alle Richtungen ausprobiert. Ich werde hinausstürzen und es verhindern. Das ist alles üble Nachrede. Der Doktor schien müde und in Eile zu sein.",
          "sentences_04": "Ich hörte es erst gestern Nacht. Wir waren jetzt im Monat März angekommen. Aber geh nur; ich hatte es vergessen. Eingebildeter Kerl mit seinem gewachsten Schnurrbart! Annes Unglück dauerte eine Woche an.",
          "sentences_05": "Tatsächlich hellte sich das Gesicht des Grafen auf. Sprich um Himmels willen mit ihr. In welch freundlichem Licht lässt ihn das erscheinen! Bring mich von meinem Weg ab. Ich hörte viele Dinge in der Hölle.",
          "sentences_06": "Ja; aber wir laden keine Leute von Rang ein. Du siehst, was er schreibt. Schweigend vor Ehrfurcht und Mitleid ging ich an ihr Bett. Ich freue mich sagen zu können, dass ich ihn nie gekannt habe. Geburtstage haben für ein vernünftiges Wesen keine Bedeutung.",
          "sentences_07": "Aber man könnte es in zwei Worten ausdrücken. Räum das Zimmer auf, sagte der Kranke mit Mühe. Er war immer noch in Sichtweite. Er zögerte; es schien fast, als hätte er vor etwas Angst. Dann trugen sie mich hinein.",
          "sentences_08": "Aber ich wurde niemals vorgestellt. Aber wir haben nur Spaß gemacht! Schau dir nun diesen dritten Namen an. Und es geschieht ihnen beiden recht. Ein gutes Glas Burgunder, das vertreibt das.",
          "sentences_09": "Und es schien ihr, als hätte Gott ihr Gebet erhört. Mein Wort, ich bewundere dich. Auch ich habe einen frommen Besuch zu machen. Sie hat versprochen, am zwanzigsten zu kommen. Ich möchte dir etwas sagen.",
          "sentences_10": "Oh, Herr, das wird Knochen brechen. Ich freue mich sehr, Sie zu sehen. Diese Frage nahm all seine geistigen Kräfte in Anspruch. Bevor ich für immer weggehe, erzähle ich ihm alles. Ich habe dir doch gesagt, dass es Mutter war.",
          "sentences_11": "Ihr seid alle gut gelaunt. Vielleicht ziehen sie sich zurück und lassen die Wachen zurück. Aber ich mag sentimentale Menschen. Unsere Kartoffelernte ist dieses Jahr sehr gut. Warum steht die Kastanie rechts?",
          "sentences_12": "Sein Zimmer war im ersten Stock. Ich hatte ein Muster in der Hand. Das Klopfen dauerte an und wurde lauter. Mögen meine Sorgen stets das Licht meiden. Wie soll ich es dann arrangieren?",
          "sentences_13": "Lies es mir einfach vor. Ich werde deinen Rat in jeder Hinsicht befolgen. Welche sterbliche Vorstellungskraft könnte das begreifen? Das Tor war wieder vom Rauch verdeckt. Nach einer Weile verließ ich ihn.",
          "sentences_14": "Ihr Atem stockte kurz. Sie sagten es mir, aber ich verstand es nicht. Was für eine Hütte das ist. Ein Freudenschrei entfuhr seinen Lippen. Er hatte den Satz offensichtlich vorher vorbereitet.",
          "sentences_15": "Sie saßen alle in ihrem Zimmer. Also so steht es also. Er wusste nicht, warum er es umarmte. Warum sprichst du nicht, Cousine? Ich habe keine Geschichte erzählt.",
          "sentences_16": "Mein Kopf schmerzt jetzt furchtbar. Um nicht jedes Wort zu sagen. Ich habe es gerade erst herausgefunden. Er versucht, etwas herauszufinden. Ich habe meine Pflicht getan.",
          "sentences_17": "Ich schätzte ihn immer. Er ist ein Betrüger und ein Schurke. Aber diese Tränen waren beiden angenehm. Sie überwand ihre Ängste und sprach. Oh, er konnte mich an der Tür nicht belauschen.",
          "sentences_18": "Wie hätte ich es noch direkter sagen können? Sie erinnerte sich an ihren Schwur. Mein Königreich für einen Drink! Haben sie das kleine Mädchen und den Jungen erwischt? Dann gab sie ihm das trockene Brot.",
          "sentences_19": "Deine Schwester ist der Regierung zugetan. Man spritzte Wasser in sein Gesicht. Die ungeschickten Dinge sind teuer. Er sprang auf und setzte sich aufs Sofa. Woher kennst du sie?",
          "sentences_20": "Ich konnte nie in meinem Leben ein Rätsel lösen. Ihr Gesichtsausdruck war kalt. Außerdem, was könnte dir überhaupt passieren? Erlauben Sie mir, Ihnen einen Rat zu geben. Das muss sofort gestoppt werden."
        },
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
