import os
import time
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, GRAD_CLIP, WARMUP_STEPS,
    SAVE_EVERY, LOG_EVERY, DEVICE, CHECKPOINT_DIR, D_MODEL, N_HEADS,
    N_KV_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN, WEIGHT_DECAY,
    TOKENIZER_TYPE, BPE_VOCAB_SIZE, TIE_WEIGHTS, RMS_NORM_EPS,
)
from tokenizer import CharTokenizer, BPETokenizer, LyricDataset
from model import RoastLyricModel
from scraper import load_all_lyrics


def get_scheduler(optimizer, warmup_steps, d_model, last_epoch=-1):
    def lr_lambda(step):
        if step == 0:
            step = 1
        return min(step ** -0.5, step * (warmup_steps ** -1.5))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    p.mul_(1 - lr * wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_nesterov = g + momentum * buf

                if g_nesterov.dim() >= 2 and min(g_nesterov.shape) > 1:
                    u, s, vt = torch.linalg.svd(g_nesterov.float(), full_matrices=False)
                    g_ortho = u @ vt
                else:
                    g_ortho = g_nesterov

                p.add_(g_ortho.to(p.dtype), alpha=-lr)

        return loss


def build_optimizer(model, use_muon=True):
    if not use_muon:
        return AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98),
                     eps=1e-9, weight_decay=WEIGHT_DECAY)

    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if p.ndim == 2 and "embedding" not in name and "embed" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    if not muon_params:
        return AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98),
                     eps=1e-9, weight_decay=WEIGHT_DECAY)

    opt_muon = Muon(muon_params, lr=0.02, momentum=0.95, weight_decay=0.0)
    opt_adamw = AdamW(adamw_params, lr=1.0, betas=(0.9, 0.98), eps=1e-9,
                      weight_decay=WEIGHT_DECAY)

    return [opt_muon, opt_adamw]


def optimizer_zero_grad(optimizers):
    for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
        opt.zero_grad()


def optimizer_step(optimizers):
    for opt in optimizers if isinstance(optimizers, list) else [optimizers]:
        opt.step()


def optimizer_state_dict(optimizers):
    if isinstance(optimizers, list):
        return [opt.state_dict() for opt in optimizers]
    return optimizers.state_dict()


def optimizer_load_state_dict(optimizers, state):
    if isinstance(optimizers, list):
        for opt, sd in zip(optimizers, state):
            try:
                opt.load_state_dict(sd)
            except Exception:
                pass
    else:
        try:
            optimizers.load_state_dict(state)
        except Exception:
            pass


def get_or_build_tokenizer(texts):
    bpe_path = os.path.join(CHECKPOINT_DIR, "tokenizer.json")
    char_path = os.path.join(CHECKPOINT_DIR, "tokenizer.pkl")

    if TOKENIZER_TYPE == "bpe" and os.path.exists(bpe_path):
        print(f"[*] Loading existing BPE tokenizer from {bpe_path}")
        tok = BPETokenizer.load(bpe_path)
        return tok, "bpe"
    if TOKENIZER_TYPE == "char" and os.path.exists(char_path):
        print(f"[*] Loading existing char tokenizer from {char_path}")
        tok = CharTokenizer.load(char_path)
        return tok, "char"

    print(f"[*] Building {TOKENIZER_TYPE} tokenizer from {len(texts)} songs...")
    if TOKENIZER_TYPE == "bpe":
        tok = BPETokenizer()
        tok.train(texts, vocab_size=BPE_VOCAB_SIZE)
        tok.save(bpe_path)
        return tok, "bpe"
    else:
        tok = CharTokenizer()
        tok.fit(texts)
        tok.save(char_path)
        return tok, "char"


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("[*] Loading lyrics...")
    texts = load_all_lyrics()
    if not texts:
        print("[!] No lyrics found in data/. Run scraper first!")
        return
    total_chars = sum(len(t) for t in texts)
    print(f"[*] Loaded {len(texts)} songs, total chars: {total_chars:,}")

    tokenizer, tok_type = get_or_build_tokenizer(texts)
    print(f"[*] Tokenizer type: {tok_type}, vocab_size: {tokenizer.vocab_size}")

    print("[*] Creating dataset...")
    dataset = LyricDataset(texts, tokenizer, seq_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"[*] Dataset size: {len(dataset)} sequences of {MAX_SEQ_LEN} tokens")
    total_tokens = len(dataset.tokens)
    print(f"[*] Total tokens in corpus: {total_tokens:,}")

    print("[*] Building model (RoPE + GQA + SwiGLU + RMSNorm)...")
    model = RoastLyricModel(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS,
        n_layers=N_LAYERS, d_ff=D_FF, max_len=MAX_SEQ_LEN,
        tie_weights=TIE_WEIGHTS,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Model parameters: {total_params:,} (trainable: {total_trainable:,})")
    print(f"[*] Architecture: d_model={D_MODEL}, n_layers={N_LAYERS}, "
          f"n_heads={N_HEADS}, n_kv_heads={N_KV_HEADS}, d_ff={D_FF}, "
          f"max_len={MAX_SEQ_LEN}, tie_weights={TIE_WEIGHTS}")

    use_muon = DEVICE == "cuda" and os.environ.get("ROAST_MUON", "0") == "1"
    optimizers = build_optimizer(model, use_muon=use_muon)
    opt_type = "Muon (2D) + AdamW (1D)" if use_muon else "AdamW"
    print(f"[*] Optimizer: {opt_type}")

    pad_id = getattr(tokenizer, "pad_id", 0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")

    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(best_path):
        print(f"[*] Resuming from {best_path}...")
        try:
            ckpt = torch.load(best_path, map_location=DEVICE, weights_only=True)
            saved_vocab = ckpt.get("vocab_size")
            if saved_vocab and saved_vocab != tokenizer.vocab_size:
                print(f"[!] Vocab mismatch: checkpoint={saved_vocab}, tokenizer={tokenizer.vocab_size}")
                print("[!] Starting from scratch.")
            else:
                model.load_state_dict(ckpt["model_state_dict"])
                optimizer_load_state_dict(optimizers, ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_loss = ckpt.get("loss", float("inf"))
                global_step = ckpt.get("global_step", start_epoch * len(loader))
                print(f"    Resumed at epoch {ckpt.get('epoch', 0)}, loss={best_loss:.4f}")
        except Exception as e:
            print(f"[!] Could not resume: {e}")
            print("[!] Starting from scratch.")

    if isinstance(optimizers, list):
        scheduler = get_scheduler(optimizers[1], WARMUP_STEPS, d_model=D_MODEL,
                                  last_epoch=global_step - 1 if global_step > 0 else -1)
    else:
        scheduler = get_scheduler(optimizers, WARMUP_STEPS, d_model=D_MODEL,
                                  last_epoch=global_step - 1 if global_step > 0 else -1)

    print(f"\n[*] Training on {DEVICE}")
    print(f"[*] Epochs {start_epoch}-{NUM_EPOCHS}, {len(loader)} batches/epoch")
    print("=" * 60)

    sample_prompts = ["I", "[Verse 1]", "[Chorus]", "Love", "Never"]

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (x, y) in enumerate(loader, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer_zero_grad(optimizers)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer_step(optimizers)
            if not isinstance(optimizers, list):
                scheduler.step()

            global_step += 1
            epoch_loss += loss.item()

            if batch_idx % LOG_EVERY == 0:
                lr = scheduler.get_last_lr()[0] if not isinstance(optimizers, list) else 0.02
                print(
                    f"  Epoch {epoch:3d} | Batch {batch_idx:5d}/{len(loader)} | "
                    f"Loss {loss.item():.4f} | LR {lr:.2e}"
                )

        avg_loss = epoch_loss / max(1, len(loader))
        elapsed = time.time() - start_time
        lr = scheduler.get_last_lr()[0] if not isinstance(optimizers, list) else 0.02
        try:
            ppl = math.exp(avg_loss)
        except OverflowError:
            ppl = float("inf")
        print(
            f"[Epoch {epoch:3d}] Avg Loss: {avg_loss:.4f} | "
            f"Perplexity: {ppl:.2f} | "
            f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_state_dict(optimizers),
                "loss": best_loss,
                "global_step": global_step,
                "vocab_size": tokenizer.vocab_size,
                "tokenizer_type": tok_type,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_kv_heads": N_KV_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "max_len": MAX_SEQ_LEN,
                "tie_weights": TIE_WEIGHTS,
            }, best_path)
            print(f"  -> Saved best model (loss={best_loss:.4f})")

        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_state_dict(optimizers),
                "loss": avg_loss,
                "global_step": global_step,
                "vocab_size": tokenizer.vocab_size,
                "tokenizer_type": tok_type,
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_kv_heads": N_KV_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "max_len": MAX_SEQ_LEN,
                "tie_weights": TIE_WEIGHTS,
            }, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

        if epoch >= 2:
            _show_sample(model, tokenizer, sample_prompts)

    print("\n[*] Training complete!")


def _show_sample(model, tokenizer, prompts):
    model.eval()
    print("\n--- Samples ---")
    with torch.no_grad():
        for prompt in prompts:
            try:
                ids = model.generate(
                    tokenizer, prompt=prompt, max_len=120,
                    temperature=0.85, top_k=60, top_p=0.92,
                )
                if hasattr(tokenizer, "decode_with_sections"):
                    text = tokenizer.decode_with_sections(ids)
                else:
                    text = tokenizer.decode(ids)
            except Exception as e:
                text = f"<error: {e}>"
            display_prompt = prompt or "<none>"
            print(f"\n  Prompt: {display_prompt!r}")
            print(f"  {text[:300]}")
    print("--- end samples ---\n")
    model.train()


if __name__ == "__main__":
    train()
