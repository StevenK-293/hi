import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, MAX_SEQ_LEN, DEVICE,
    REPETITION_PENALTY, NO_REPEAT_NGRAM,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN + 64, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        Q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=DROPOUT):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class LyricTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_ff=D_FF, max_len=MAX_SEQ_LEN,
                 dropout=DROPOUT):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.max_len = max_len
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return ~mask

    def forward(self, x):
        B, T = x.shape
        device = x.device
        tok_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(tok_emb)
        mask = self._create_causal_mask(T, device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, tokenizer, prompt="", max_len=300,
                 temperature=0.85, top_k=60, top_p=0.92,
                 repetition_penalty=REPETITION_PENALTY,
                 no_repeat_ngram=NO_REPEAT_NGRAM):
        self.eval()

        prompt_ids = tokenizer.encode(prompt, add_special=False) if prompt else []
        if not prompt_ids:
            eos_id = getattr(tokenizer, "eos_id", 2)
            prompt_ids = [eos_id]

        x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
        generated = list(prompt_ids)

        eos_id = getattr(tokenizer, "eos_id", 2)

        with torch.no_grad():
            for _ in range(max_len):
                if x.size(1) > self.max_len:
                    x = x[:, -self.max_len:]
                logits = self.forward(x)
                next_logits = logits[0, -1, :].clone()

                if repetition_penalty and repetition_penalty != 1.0:
                    seen = set(generated[-256:])
                    for tok_id in seen:
                        if next_logits[tok_id] > 0:
                            next_logits[tok_id] /= repetition_penalty
                        else:
                            next_logits[tok_id] *= repetition_penalty

                if no_repeat_ngram and no_repeat_ngram > 0 and len(generated) >= no_repeat_ngram - 1:
                    ngram_prefix = tuple(generated[-(no_repeat_ngram - 1):])
                    banned = set()
                    for i in range(len(generated) - no_repeat_ngram + 1):
                        if tuple(generated[i:i + no_repeat_ngram - 1]) == ngram_prefix:
                            banned.add(generated[i + no_repeat_ngram - 1])
                    for b in banned:
                        next_logits[b] = float("-inf")

                next_logits = next_logits / max(temperature, 1e-5)
                next_logits = self._top_k_top_p_filtering(next_logits, top_k, top_p)

                probs = F.softmax(next_logits, dim=-1)
                if torch.isnan(probs).any() or probs.sum() == 0:
                    next_id = eos_id
                else:
                    next_id = torch.multinomial(probs, num_samples=1).item()

                if next_id == eos_id and len(generated) < 8:
                    probs_ = probs.clone()
                    probs_[eos_id] = 0
                    if probs_.sum() > 0:
                        next_id = torch.multinomial(probs_, num_samples=1).item()

                generated.append(next_id)
                x = torch.cat([x, torch.tensor([[next_id]], device=DEVICE)], dim=1)

                if next_id == eos_id:
                    break

        return generated

    def _top_k_top_p_filtering(self, logits, top_k, top_p):
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < values[-1], float("-inf"))
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            return sorted_logits.scatter(-1, sorted_indices, sorted_logits)
        return logits
