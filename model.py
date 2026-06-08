import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    D_MODEL, N_HEADS, N_KV_HEADS, N_LAYERS, D_FF, DROPOUT,
    MAX_SEQ_LEN, DEVICE, ROPE_THETA, RMS_NORM_EPS, TIE_WEIGHTS,
    REPETITION_PENALTY, NO_REPEAT_NGRAM,
)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_rope_freqs(dim, max_len, theta=ROPE_THETA):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    B, H, T, D = x.shape
    D2 = D // 2
    x_pair = x.float().reshape(B, H, T, 2, D2)
    x0, x1 = x_pair.unbind(-2)
    cos = cos[:T, :D2].unsqueeze(0).unsqueeze(0)
    sin = sin[:T, :D2].unsqueeze(0).unsqueeze(0)
    rotated_x0 = x0 * cos - x1 * sin
    rotated_x1 = x0 * sin + x1 * cos
    return torch.stack([rotated_x0, rotated_x1], dim=-2).reshape(B, H, T, D).type_as(x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=DROPOUT):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        B, T, D = x.shape
        Q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            K = K.repeat_interleave(repeat, dim=1)
            V = V.repeat_interleave(repeat, dim=1)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ V
        return self.w_o(out.transpose(1, 2).contiguous().view(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=DROPOUT):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout=DROPOUT):
        super().__init__()
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        self.norm1 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff, dropout)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class RoastLyricModel(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
                 n_kv_heads=N_KV_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
                 max_len=MAX_SEQ_LEN, dropout=DROPOUT,
                 tie_weights=TIE_WEIGHTS):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        self.d_model = d_model
        self.max_len = max_len
        self.vocab_size = vocab_size

        cos, sin = precompute_rope_freqs(d_model // n_heads, max_len + 64)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def _create_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return ~mask

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        mask = self._create_causal_mask(T, x.device)
        cos = self.rope_cos
        sin = self.rope_sin
        if cos.shape[0] < T:
            cos, sin = precompute_rope_freqs(
                self.d_model // self.blocks[0].attn.head_dim, max(T, self.max_len + 64),
            )
            cos, sin = cos.to(x.device), sin.to(x.device)
            self.rope_cos = cos
            self.rope_sin = sin

        x = tok_emb
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
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
        pad_id = getattr(tokenizer, "pad_id", 0)
        sos_id = getattr(tokenizer, "sos_id", 1)

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
