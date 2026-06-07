import os
import re
import sys
import argparse
import torch
from config import (
    DEVICE, CHECKPOINT_DIR, TEMPERATURE, TOP_K, TOP_P, MAX_GEN_LEN,
    REPETITION_PENALTY, NO_REPEAT_NGRAM,
)
from tokenizer import load_tokenizer
from model import LyricTransformer


def _strip_artist_brackets(text):
    text = re.sub(r"\[([^\]]+)\s*:[^\]]*\]", r"[\1]", text)
    text = re.sub(r"\[(\w+)\s*&\s*[^\]]*\]", r"[\1]", text)
    return text


def load_model(checkpoint_path=None):
    if checkpoint_path is None:
        best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        if os.path.exists(best_path):
            checkpoint_path = best_path
        else:
            checkpoints = [f for f in os.listdir(CHECKPOINT_DIR)
                           if f.endswith(".pt") and f != "tokenizer.pkl"]
            if not checkpoints:
                print("[!] No model checkpoints found!")
                print("    Train a model first: python train.py")
                sys.exit(1)
            checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)))
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoints[-1])

    try:
        tokenizer, tok_type = load_tokenizer(CHECKPOINT_DIR)
    except FileNotFoundError as e:
        print(f"[!] {e}")
        sys.exit(1)
    print(f"[*] Loaded {tok_type} tokenizer: vocab_size={tokenizer.vocab_size}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    print(f"[*] Loaded checkpoint: {checkpoint_path} (epoch={checkpoint.get('epoch', '?')}, "
          f"loss={checkpoint.get('loss', float('nan')):.4f})")

    d_model = checkpoint.get("d_model", 256)
    n_heads = checkpoint.get("n_heads", 8)
    n_layers = checkpoint.get("n_layers", 6)
    d_ff = checkpoint.get("d_ff", 512)
    max_len = checkpoint.get("max_len", 768)

    model = LyricTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_ff=d_ff, max_len=max_len,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model: {n_params:,} params, d_model={d_model}, n_layers={n_layers}")

    return model, tokenizer, tok_type


def generate_lyrics(
    model, tokenizer, prompt="", count=1, max_len=MAX_GEN_LEN,
    temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY, no_repeat_ngram=NO_REPEAT_NGRAM,
):
    results = []
    for i in range(count):
        ids = model.generate(
            tokenizer, prompt=prompt,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram=no_repeat_ngram,
        )
        if hasattr(tokenizer, "decode_with_sections"):
            text = tokenizer.decode_with_sections(ids)
        else:
            text = tokenizer.decode(ids)
        text = _strip_artist_brackets(text)
        results.append(text)
    return results


def interactive_mode(model, tokenizer):
    print("\n=== Lyric Generation Interactive Mode ===")
    print("Enter a prompt (or 'quit' to exit, 'help' for options)")
    print("-" * 50)

    settings = {
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "max_len": MAX_GEN_LEN,
        "repetition_penalty": REPETITION_PENALTY,
        "no_repeat_ngram": NO_REPEAT_NGRAM,
    }

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if prompt.lower() in ("help", "h"):
            print("Options:")
            print("  quit/exit/q              - Exit")
            print("  set temp <N>             - Set temperature (default: 0.85)")
            print("  set topk <N>             - Set top-k (default: 60)")
            print("  set topp <N>             - Set top-p (default: 0.92)")
            print("  set len <N>              - Set max generation length (default: 400)")
            print("  set rep <N>              - Repetition penalty (default: 1.15)")
            print("  set ngram <N>            - No-repeat ngram size (default: 4)")
            print("  show                     - Show current settings")
            print("  reset                    - Reset settings to defaults")
            print("  song <theme>             - Generate a full song structure")
            print("\nTips:")
            print("  - Start with a section tag like '[Chorus]' or '[Verse 1]'")
            print("    to condition on a specific song part.")
            print("  - e.g. '[Chorus] We own the night' or '[Hook] Yeah'")
            continue
        if prompt.lower() == "show":
            print(f"  temperature={settings['temperature']}  top_k={settings['top_k']}  "
                  f"top_p={settings['top_p']}")
            print(f"  max_len={settings['max_len']}  rep_penalty={settings['repetition_penalty']}  "
                  f"no_repeat_ngram={settings['no_repeat_ngram']}")
            continue
        if prompt.lower() == "reset":
            settings = {
                "temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P,
                "max_len": MAX_GEN_LEN, "repetition_penalty": REPETITION_PENALTY,
                "no_repeat_ngram": NO_REPEAT_NGRAM,
            }
            print("  settings reset to defaults")
            continue
        if prompt.lower().startswith("song "):
            theme = prompt[5:].strip()
            if not theme:
                theme = "the night"
            print(f"\n[*] Generating song about: {theme}")
            sections = [
                ("[Intro]", f" {theme}"),
                ("[Verse 1]", f" {theme}"),
                ("[Chorus]", f" {theme}"),
                ("[Verse 2]", f" {theme}"),
                ("[Chorus]", f" {theme}"),
                ("[Outro]", f" {theme}"),
            ]
            full = ""
            for tag, seed in sections:
                sub = generate_lyrics(
                    model, tokenizer, prompt=tag + seed, count=1,
                    max_len=200, temperature=settings["temperature"],
                    top_k=settings["top_k"], top_p=settings["top_p"],
                    repetition_penalty=settings["repetition_penalty"],
                    no_repeat_ngram=settings["no_repeat_ngram"],
                )[0]
                full += sub + "\n\n"
                print(f"\n{'='*50}\n{sub}\n{'='*50}")
            continue
        if prompt.startswith("set "):
            parts = prompt.split()
            if len(parts) != 3:
                print("  usage: set <temp|topk|topp|len|rep|ngram> <value>")
                continue
            _, key, val = parts
            key_map = {
                "temp": "temperature", "topk": "top_k", "topp": "top_p",
                "len": "max_len", "rep": "repetition_penalty", "ngram": "no_repeat_ngram",
            }
            if key not in key_map:
                print(f"  unknown setting '{key}'. try: temp, topk, topp, len, rep, ngram")
                continue
            try:
                v = float(val)
                if key_map[key] in ("max_len", "no_repeat_ngram", "top_k"):
                    v = int(v)
                settings[key_map[key]] = v
            except ValueError:
                print(f"  '{val}' is not a number")
                continue
            print(f"  {key_map[key]} = {settings[key_map[key]]}")
            continue

        lyrics = generate_lyrics(
            model, tokenizer, prompt=prompt,
            max_len=settings["max_len"],
            temperature=settings["temperature"],
            top_k=settings["top_k"], top_p=settings["top_p"],
            repetition_penalty=settings["repetition_penalty"],
            no_repeat_ngram=settings["no_repeat_ngram"],
        )
        print("\n" + "=" * 50)
        print(lyrics[0])
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Generate lyrics with trained transformer")
    parser.add_argument("prompt", nargs="?", default="", help="Prompt text to start generation")
    parser.add_argument("-n", "--count", type=int, default=1, help="Number of lyrics to generate")
    parser.add_argument("--max-len", type=int, default=MAX_GEN_LEN, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--rep", type=float, default=REPETITION_PENALTY,
                        help="Repetition penalty (1.0 = off)")
    parser.add_argument("--ngram", type=int, default=NO_REPEAT_NGRAM,
                        help="No-repeat n-gram size (0 = off)")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--song", metavar="THEME", default=None,
                        help="Generate a full multi-section song about a theme")
    args = parser.parse_args()

    if not os.path.isdir(CHECKPOINT_DIR) or not os.listdir(CHECKPOINT_DIR):
        print("[!] No checkpoints found. Train the model first:")
        print("    python scraper.py <artist_or_query>")
        print("    python train.py")
        sys.exit(1)

    model, tokenizer, tok_type = load_model(args.checkpoint)

    if args.song is not None:
        theme = args.song if args.song else "the night"
        print(f"\n[*] Generating song about: {theme}")
        sections = [
            "[Intro]", "[Verse 1]", "[Chorus]", "[Verse 2]",
            "[Chorus]", "[Bridge]", "[Outro]",
        ]
        for tag in sections:
            sub = generate_lyrics(
                model, tokenizer, prompt=f"{tag} {theme}", count=1,
                max_len=args.max_len, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.rep, no_repeat_ngram=args.ngram,
            )[0]
            print(f"\n{'='*50}\n{sub}\n{'='*50}")
        return

    if args.interactive or not args.prompt:
        interactive_mode(model, tokenizer)
        return

    lyrics_list = generate_lyrics(
        model, tokenizer, prompt=args.prompt,
        count=args.count, max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.rep, no_repeat_ngram=args.ngram,
    )
    for i, lyrics in enumerate(lyrics_list, 1):
        if args.count > 1:
            print(f"\n=== Generation {i} ===")
        print(lyrics)


if __name__ == "__main__":
    main()
