import os
import re
import json
import pickle
from config import (
    SPECIAL_TOKENS, SECTION_TOKENS,
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
    MAX_SEQ_LEN, DATA_DIR,
    TOKENIZER_TYPE, BPE_VOCAB_SIZE,
    TOKENIZER_PATH, TOKENIZER_META_PATH,
)

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    HAS_HF_TOKENIZERS = True
except ImportError:
    HAS_HF_TOKENIZERS = False


class CharTokenizer:
    def __init__(self, chars=None):
        self.word2idx = {}
        self.idx2word = {}
        if chars is not None:
            self._build_from_chars(chars)

    def _build_from_chars(self, chars):
        sorted_chars = sorted(set(chars))
        all_tokens = SPECIAL_TOKENS + sorted_chars
        self.word2idx = {c: i for i, c in enumerate(all_tokens)}
        self.idx2word = {i: c for c, i in self.word2idx.items()}

    def fit(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        self._build_from_chars(chars)

    def encode(self, text, add_special=True):
        ids = []
        if add_special:
            ids.append(self.word2idx.get(SOS_TOKEN, PAD_IDX))
        for ch in text:
            ids.append(self.word2idx.get(ch, self.word2idx.get(UNK_TOKEN, PAD_IDX)))
        if add_special:
            ids.append(self.word2idx.get(EOS_TOKEN, PAD_IDX))
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            ch = self.idx2word.get(i, UNK_TOKEN)
            if ch in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                continue
            chars.append(ch)
        result = "".join(chars)
        result = result.encode("ascii", errors="ignore").decode("ascii")
        return result

    @property
    def vocab_size(self):
        return len(self.word2idx)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        tok = cls()
        tok.word2idx = data["word2idx"]
        tok.idx2word = data["idx2word"]
        return tok


class BPETokenizer:
    def __init__(self):
        self._tk = None
        self._vocab = None
        self._token_to_id = {}
        self._id_to_token = {}
        self.pad_id = PAD_IDX
        self.sos_id = SOS_IDX
        self.eos_id = EOS_IDX
        self.unk_id = UNK_IDX
        self._section_token_ids = set()

    @staticmethod
    def _protect_sections(text):
        for tag in SECTION_TOKENS:
            text = text.replace(tag, f" {tag} ")
        text = re.sub(r"[ \t]+", " ", text)
        return text

    def train(self, texts, vocab_size=BPE_VOCAB_SIZE):
        if not HAS_HF_TOKENIZERS:
            raise ImportError("Install `tokenizers` (pip install tokenizers)")

        marked = [self._protect_sections(t) for t in texts]

        tk = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
            min_frequency=2,
        )
        tk.train_from_iterator(marked, trainer=trainer)
        tk.post_processor = processors.ByteLevel(trim_offsets=False)

        self._tk = tk
        self._vocab = tk.get_vocab()
        self._token_to_id = dict(self._vocab)
        self._id_to_token = {i: t for t, i in self._vocab.items()}

        self.pad_id = self._vocab.get(PAD_TOKEN, PAD_IDX)
        self.sos_id = self._vocab.get(SOS_TOKEN, SOS_IDX)
        self.eos_id = self._vocab.get(EOS_TOKEN, EOS_IDX)
        self.unk_id = self._vocab.get(UNK_TOKEN, UNK_IDX)
        self._section_token_ids = {self._vocab[t] for t in SECTION_TOKENS if t in self._vocab}

    def encode(self, text, add_special=True):
        text = self._protect_sections(text)
        ids = self._tk.encode(text).ids
        if add_special:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if skip_special:
            filtered = [
                i for i in ids
                if i not in (self.pad_id, self.sos_id, self.eos_id, self.unk_id)
                and i not in self._section_token_ids
            ]
            text = self._tk.decode(filtered, skip_special_tokens=False)
        else:
            text = self._tk.decode(list(ids), skip_special_tokens=False)
        text = text.replace("\u0120", " ").replace("\u010a", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def decode_with_sections(self, ids):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        text = self._tk.decode(list(ids), skip_special_tokens=False)
        drop = {PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN}
        for tag in drop:
            text = text.replace(tag, "")
        for tag in SECTION_TOKENS:
            text = text.replace(f" {tag} ", f"\n\n{tag}\n")
            text = text.replace(tag, f"\n{tag}\n")
        text = text.replace("\u0120", " ").replace("\u010a", "\n")
        text = text.replace("\u0120", " ").replace("\u010a", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @property
    def vocab_size(self):
        return len(self._vocab) if self._vocab else 0

    def save(self, path):
        if path.endswith(".json"):
            self._tk.save(path)
            meta = {
                "type": "bpe",
                "pad_id": self.pad_id,
                "sos_id": self.sos_id,
                "eos_id": self.eos_id,
                "unk_id": self.unk_id,
                "section_token_ids": list(self._section_token_ids),
                "vocab_size": self.vocab_size,
            }
            with open(path + ".meta.json", "w") as f:
                json.dump(meta, f)
        else:
            self._tk.save(path + ".json")
            with open(path + ".meta.json", "w") as f:
                json.dump({
                    "type": "bpe",
                    "pad_id": self.pad_id,
                    "sos_id": self.sos_id,
                    "eos_id": self.eos_id,
                    "unk_id": self.unk_id,
                    "section_token_ids": list(self._section_token_ids),
                    "vocab_size": self.vocab_size,
                }, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        json_path = path if path.endswith(".json") else path + ".json"
        meta_path = json_path + ".meta.json"
        obj._tk = Tokenizer.from_file(json_path)
        obj._vocab = obj._tk.get_vocab()
        obj._token_to_id = dict(obj._vocab)
        obj._id_to_token = {i: t for t, i in obj._vocab.items()}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            obj.pad_id = meta.get("pad_id", PAD_IDX)
            obj.sos_id = meta.get("sos_id", SOS_IDX)
            obj.eos_id = meta.get("eos_id", EOS_IDX)
            obj.unk_id = meta.get("unk_id", UNK_IDX)
            obj._section_token_ids = set(meta.get("section_token_ids", []))
        return obj


class LyricDataset:
    def __init__(self, texts, tokenizer, seq_len=MAX_SEQ_LEN, stride=256):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride

        if isinstance(tokenizer, BPETokenizer):
            separator = "\n\n"
        else:
            separator = "\n\n---\n\n"
        full = separator.join(texts)
        self.tokens = tokenizer.encode(full, add_special=True)
        self.indices = list(range(0, max(1, len(self.tokens) - self.seq_len - 1), stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        import torch
        start = self.indices[idx]
        chunk = self.tokens[start: start + self.seq_len + 1]
        if len(chunk) < 2:
            chunk = chunk + [self.tokenizer.pad_id] * (2 - len(chunk))
        x = chunk[:-1]
        y = chunk[1:]
        if len(x) < self.seq_len:
            pad_len = self.seq_len - len(x)
            x = x + [self.tokenizer.pad_id] * pad_len
            y = y + [self.tokenizer.pad_id] * pad_len
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def detect_tokenizer_type(checkpoint_dir=os.path.dirname(TOKENIZER_PATH)):
    meta_path = os.path.join(checkpoint_dir, "tokenizer.meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("type", "char")
        except Exception:
            pass
    if os.path.exists(TOKENIZER_PATH + ".json"):
        return "bpe"
    if os.path.exists(TOKENIZER_PATH):
        try:
            with open(TOKENIZER_PATH, "rb") as f:
                pickle.load(f)
            return "char"
        except Exception:
            pass
    return TOKENIZER_TYPE


def load_tokenizer(checkpoint_dir=os.path.dirname(TOKENIZER_PATH)):
    tok_type = detect_tokenizer_type(checkpoint_dir)
    if tok_type == "bpe":
        path = os.path.join(checkpoint_dir, "tokenizer.json")
        if os.path.exists(path):
            return BPETokenizer.load(path), "bpe"
    pkl = os.path.join(checkpoint_dir, "tokenizer.pkl")
    if os.path.exists(pkl):
        return CharTokenizer.load(pkl), "char"
    raise FileNotFoundError(f"No tokenizer found in {checkpoint_dir}")


def build_tokenizer_from_data(tokenizer_type=TOKENIZER_TYPE, vocab_size=BPE_VOCAB_SIZE):
    from scraper import load_all_lyrics
    texts = load_all_lyrics()
    if not texts:
        raise RuntimeError("No lyrics found in data/. Run scraper first.")

    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=vocab_size)
    else:
        tokenizer = CharTokenizer()
        tokenizer.fit(texts)

    print(f"[*] Tokenizer built ({tokenizer_type}): vocab_size={tokenizer.vocab_size}")
    return tokenizer, texts
