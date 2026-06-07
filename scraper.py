import requests
from bs4 import BeautifulSoup
import time
import json
import re
import os
from config import GENIUS_API_SEARCH, REQUEST_DELAY, REQUEST_TIMEOUT, MAX_SONGS_PER_ARTIST, DATA_DIR

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _request_with_delay(url, params=None):
    time.sleep(REQUEST_DELAY)
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        print(f"[!] Request failed: {url} — {e}")
        return None


def search_songs(query, page=1, per_page=10):
    params = {"q": query, "page": page, "per_page": per_page}
    resp = _request_with_delay(GENIUS_API_SEARCH, params=params)
    if not resp:
        return []
    try:
        data = resp.json()
        hits = []
        for section in data.get("response", {}).get("sections", []):
            for hit in section.get("hits", []):
                if hit.get("type") == "song":
                    hits.append(hit["result"])
        return hits
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[!] Failed to parse search results: {e}")
        return []


_SECTION_HEADER_RE = re.compile(r"^\[([^\]]+)\]\s*$")


def normalize_section_tags(text):
    section_map = {
        "intro": "[Intro]",
        "verse": "[Verse 1]",
        "verse 1": "[Verse 1]",
        "verse 2": "[Verse 2]",
        "verse 3": "[Verse 3]",
        "verse 4": "[Verse 4]",
        "verse 5": "[Verse 5]",
        "chorus": "[Chorus]",
        "hook": "[Hook]",
        "refrain": "[Refrain]",
        "pre-chorus": "[Pre-Chorus]",
        "pre chorus": "[Pre-Chorus]",
        "prechorus": "[Pre-Chorus]",
        "post-chorus": "[Post-Chorus]",
        "post chorus": "[Post-Chorus]",
        "postchorus": "[Post-Chorus]",
        "bridge": "[Bridge]",
        "outro": "[Outro]",
        "interlude": "[Interlude]",
        "skit": "[Skit]",
        "end": "[End]",
    }

    out_lines = []
    for line in text.split("\n"):
        m = _SECTION_HEADER_RE.match(line.strip())
        if not m:
            out_lines.append(line)
            continue
        inner = m.group(1).strip()
        if not inner:
            out_lines.append(line)
            continue

        primary = inner.split(",")[0].split("&")[0].split("/")[0].strip()
        primary = primary.split(":")[0].strip().lower()

        if primary.startswith("part"):
            out_lines.append("[Verse 1]")
            continue

        if primary.startswith("verse"):
            nums = re.findall(r"\d+", primary)
            if nums:
                n = int(nums[0])
                if n > 5:
                    n = 5
                out_lines.append(f"[Verse {n}]")
            else:
                out_lines.append("[Verse 1]")
            continue

        normalized = section_map.get(primary)
        if normalized:
            out_lines.append(normalized)
        else:
            out_lines.append(line)

    return "\n".join(out_lines)


def clean_lyrics(raw_text):
    if not raw_text:
        return None
    text = raw_text
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\d+\s*ContributorsTranslations\S*", "", text)
    text = re.sub(r"Read More\s*", "", text)
    text = re.sub(r"See .+? LiveGet tickets as low as \$\d+", "", text)
    text = re.sub(r"You might also like", "", text)
    text = re.sub(r"Embed", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^\[.*\]", stripped):
            start = i
            break
    text = "\n".join(lines[start:]).strip()
    if not text:
        return None
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if re.match(r"^\d+\s*$", line.strip()):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = normalize_section_tags(text)
    return text if text else None


def normalize_existing_files():
    count = 0
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    original = f.read()
                normalized = normalize_section_tags(original)
                if normalized != original:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(normalized)
                    count += 1
            except Exception as e:
                print(f"[!] Failed to normalize {fpath}: {e}")
    return count


def get_song_lyrics(song_url):
    resp = _request_with_delay(song_url)
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    containers = soup.select("[data-lyrics-container='true']")
    if not containers:
        containers = soup.select("div.lyrics")
    if not containers:
        containers = soup.select("div[class*='Lyrics__Container']")
    if not containers:
        return None

    lines = []
    for container in containers:
        html = str(container)
        html = html.replace("<br/>", "\n").replace("<br>", "\n")
        text = BeautifulSoup(html, "html.parser").get_text()
        lines.append(text)
    full = "\n".join(lines)
    full = re.sub(r"\n{3,}", "\n\n", full).strip()
    return clean_lyrics(full)


def scrape_artist(artist_name, max_songs=MAX_SONGS_PER_ARTIST):
    print(f"[*] Searching for songs by '{artist_name}'...")
    all_songs = []
    page = 1
    while len(all_songs) < max_songs:
        results = search_songs(artist_name, page=page, per_page=20)
        if not results:
            break
        artist_songs = [s for s in results if artist_name.lower() in s.get("artist_names", "").lower()]
        all_songs.extend(artist_songs)
        page += 1
        if len(artist_songs) < 5:
            break

    all_songs = all_songs[:max_songs]
    print(f"[*] Found {len(all_songs)} songs. Scraping lyrics...")

    lyrics_collected = []
    for i, song in enumerate(all_songs, 1):
        title = song.get("title", "Unknown")
        url = song.get("url", "")
        if not url:
            continue
        print(f"  [{i}/{len(all_songs)}] {title}...", end=" ", flush=True)
        lyrics = get_song_lyrics(url)
        if lyrics:
            filepath = _save_lyrics(artist_name, title, lyrics)
            lyrics_collected.append({"title": title, "artist": artist_name, "filepath": filepath})
            print(f"saved ({len(lyrics.split())} words)")
        else:
            print("no lyrics found")

    return lyrics_collected


def scrape_top_songs(query, count=50):
    print(f"[*] Searching top songs for '{query}'...")
    all_songs = []
    page = 1
    while len(all_songs) < count:
        results = search_songs(query, page=page, per_page=20)
        if not results:
            break
        all_songs.extend(results)
        page += 1
        if len(results) < 5:
            break

    all_songs = all_songs[:count]
    print(f"[*] Found {len(all_songs)} songs. Scraping lyrics...")

    lyrics_collected = []
    for i, song in enumerate(all_songs, 1):
        title = song.get("title", "Unknown")
        artist = song.get("artist_names", "Unknown")
        url = song.get("url", "")
        if not url:
            continue
        print(f"  [{i}/{len(all_songs)}] {artist} — {title}...", end=" ", flush=True)
        lyrics = get_song_lyrics(url)
        if lyrics:
            safe_artist = _sanitize_filename(artist)
            filepath = _save_lyrics(safe_artist, title, lyrics)
            lyrics_collected.append({"title": title, "artist": artist, "filepath": filepath})
            print(f"saved ({len(lyrics.split())} words)")
        else:
            print("no lyrics found")

    return lyrics_collected


def _sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


def _save_lyrics(artist, title, lyrics):
    safe_artist = _sanitize_filename(artist)
    safe_title = _sanitize_filename(title)
    artist_dir = os.path.join(DATA_DIR, safe_artist)
    os.makedirs(artist_dir, exist_ok=True)
    filepath = os.path.join(artist_dir, f"{safe_title}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(lyrics)
    return filepath


def load_all_lyrics():
    all_text = []
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.endswith(".txt"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if text:
                        all_text.append(text)
                except Exception as e:
                    print(f"[!] Failed to read {fpath}: {e}")
    return all_text


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--normalize":
        print("[*] Normalizing existing data files (stripping ': ArtistName' from section tags)...")
        n = normalize_existing_files()
        print(f"[*] Normalized {n} files")
    elif len(sys.argv) > 1:
        query = sys.argv[1]
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        scrape_top_songs(query, count)
    else:
        print("Usage:")
        print("  python scraper.py <artist_or_query> [count]")
        print("  python scraper.py --normalize")
