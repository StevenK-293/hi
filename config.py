import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

GENIUS_BASE_URL = "https://genius.com"
GENIUS_API_SEARCH = "https://genius.com/api/search/song"
REQUEST_DELAY = 1.5
MAX_SONGS_PER_ARTIST = 30
REQUEST_TIMEOUT = 15

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SECTION_TOKENS = [
    "[Intro]", "[Verse 1]", "[Verse 2]", "[Verse 3]", "[Verse 4]", "[Verse 5]",
    "[Pre-Chorus]", "[Post-Chorus]", "[Chorus]", "[Hook]", "[Refrain]",
    "[Bridge]", "[Outro]", "[Interlude]", "[Skit]", "[End]",
]
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + SECTION_TOKENS
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

TOKENIZER_TYPE = "bpe"
BPE_VOCAB_SIZE = 4000
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.json")
TOKENIZER_META_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer.meta.json")

MAX_SEQ_LEN = 768

D_MODEL = 320
N_HEADS = 8
N_KV_HEADS = 4
N_LAYERS = 8
D_FF = 896
DROPOUT = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROPE_THETA = 10000.0
RMS_NORM_EPS = 1e-6
TIE_WEIGHTS = True

BATCH_SIZE = 12
LEARNING_RATE = 5e-4
NUM_EPOCHS = 35
GRAD_CLIP = 1.0
WARMUP_STEPS = 200
SAVE_EVERY = 5
LOG_EVERY = 100
WEIGHT_DECAY = 0.01

TEMPERATURE = 0.85
TOP_K = 60
TOP_P = 0.92
MAX_GEN_LEN = 400
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM = 4
