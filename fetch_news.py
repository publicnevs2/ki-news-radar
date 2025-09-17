#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_news.py
- Holt RSS/Podcast-Feeds
- Dedupliziert + begrenzt nach Zeit/Anzahl
- Anreicherung via Gemini (summary_ai, topics, sentiment)
- Schreibt sortiert in data.json
"""

import os
# ---- gRPC/TF-Log-Noise dämpfen (muss VOR anderen Imports passieren) ----
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import json
import html
import time
import feedparser
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

import google.generativeai as genai

# =========================
#   1) KONFIG & LIMITS
# =========================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("FEHLER: GEMINI_API_KEY nicht gefunden. Skript wird beendet.")
    raise SystemExit(1)

genai.configure(api_key=GEMINI_API_KEY)

# Generation- & Safety-Settings versionsrobust als Dicts
GENERATION_CONFIG = {"response_mime_type": "application/json"}
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Laufzeit-/Backfill-Parameter (über Env überschreibbar)
MAX_PER_FEED      = int(os.getenv("MAX_PER_FEED", "3"))
MAX_TOTAL         = int(os.getenv("MAX_TOTAL", "30"))
RECENCY_DAYS      = int(os.getenv("RECENCY_DAYS", "14"))
RUN_BUDGET_SEC    = int(os.getenv("RUN_BUDGET_SEC", "180"))
FIRST_RUN_SHALLOW = os.getenv("FIRST_RUN_SHALLOW", "true").lower() == "true"

DATA_FILE = "data.json"

# =========================
#   2) FEEDS
# =========================
RSS_FEEDS = {
    # Deutsche Podcasts
    "Heise KI Update": {"url": "https://kiupdate.podigee.io/feed/mp3", "type": "podcast"},
    "AI First": {"url": "https://feeds.captivate.fm/ai-first/", "type": "podcast"},
    "KI Inside": {"url": "https://agidomedia.podcaster.de/insideki.rss", "type": "podcast"},
    "KI>Inside": {"url": "https://anchor.fm/s/fb4ad23c/podcast/rss", "type": "podcast"},
    #"Your CoPilot": {"url": "https://podcast.yourcopilot.de/feed/mp3", "type": "podcast"},
    #"KI verstehen": {"url": "https://www.deutschlandfunk.de/ki-verstehen-102.xml", "type": "podcast"},

    # Englische News & Blogs
    "MIT Technology Review": {"url": "https://www.technologyreview.com/feed/", "type": "article"},
    #"KDnuggets": {"url": "https://www.kdnuggets.com/feed", "type": "article"},
    #"OpenAI Blog": {"url": "https://openai.com/blog/rss.xml", "type": "article"},
    #"Google AI Blog": {"url": "https://ai.googleblog.com/feeds/posts/default?alt=rss", "type": "article"},
    #"Hugging Face Blog": {"url": "https://huggingface.co/blog/feed.xml", "type": "article"},
    #"DeepMind Blog": {"url": "https://deepmind.google/discover/blog/feed/basic", "type": "article"},
    #"BAIR Blog (Berkeley AI Research)": {"url": "https://bair.berkeley.edu/blog/feed.xml", "type": "article"},
    #"AI Trends": {"url": "https://aitrends.com/feed/", "type": "article"},
    #"NVIDIA Blog (AI)": {"url": "https://blogs.nvidia.com/blog/category/ai/feed/", "type": "article"},
    #"VentureBeat AI": {"url": "https://venturebeat.com/category/ai/feed/", "type": "article"},

    # Deutsche News
    #"Heise (Thema KI)": {"url": "https://www.heise.de/thema/kuenstliche-intelligenz/rss.xml", "type": "article"},
    "t3n (Thema KI)": {"url": "https://t3n.de/tag/ki/rss", "type": "article"},
    #"ZEIT ONLINE (Digital)": {"url": "https://newsfeed.zeit.de/digital/index", "type": "article"},
    "Handelsblatt KI": {"url": "https://www.handelsblatt.com/contentexport/feed/schlagworte/10026866", "type": "article"},
    "Tagesschau (Digitales)": {"url": "https://www.tagesschau.de/xml/rss2_-_thema-digitales-101.xml", "type": "article"},
}

PROMPT = """
Du bist ein JSON-Generator. Antworte ausschließlich mit einem einzelnen JSON-Objekt ohne Erklärtext oder Markdown.
Schlüssel:
1) "summary_ai": max. 2 Sätze (Deutsch).
2) "topics": genau 3 Schlagwörter (Array aus Strings, Deutsch).
3) "sentiment": einer von ["positiv","neutral","negativ"].

Eingang:
---
Titel: {title}
Text: {content}
---
"""

# =========================
#   3) HELFER
# =========================
def load_existing_data():
    """Lädt data.json → Dict uid->item (oder leer)."""
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            result = {}
            for item in data:
                uid = item.get("uid") or item.get("link")
                if uid:
                    result[uid] = item
            return result
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def _clean_html(text: str) -> str:
    """HTML-Tags entfernen, Entities unescapen, Whitespaces normalisieren."""
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+?>", "", text)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()

def _iso_from_struct_time(st) -> str:
    """feedparser struct_time → UTC ISO 8601."""
    if not st:
        return datetime.now(timezone.utc).isoformat()
    dt = datetime.fromtimestamp(time.mktime(st), tz=timezone.utc)
    return dt.isoformat()

def _find_audio_url(entry) -> str:
    """Audio-URL in Enclosures oder Links mit rel='enclosure' suchen."""
    try:
        if hasattr(entry, "enclosures"):
            for enc in entry.enclosures:
                t = enc.get("type", "") if isinstance(enc, dict) else getattr(enc, "type", "")
                href = enc.get("href", "") if isinstance(enc, dict) else getattr(enc, "href", "")
                if "audio" in (t or "") and href:
                    return href
    except Exception:
        pass
    try:
        for l in getattr(entry, "links", []):
            if l.get("rel") == "enclosure" and "audio" in l.get("type", ""):
                return l.get("href", "")
    except Exception:
        pass
    return ""

def _make_uid(entry) -> str:
    """UID aus id/guid/link bilden (Fallback mit Titel+Zeit)."""
    uid = getattr(entry, "id", None) or getattr(entry, "guid", None) or getattr(entry, "link", None)
    return uid or f"no-id::{getattr(entry, 'title', '')[:50]}::{time.time()}"

def _is_recent(iso_str: str, days: int) -> bool:
    """True, wenn published innerhalb der letzten X Tage liegt (UTC)."""
    if days <= 0:
        return True
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except Exception:
        return True
    return dt >= datetime.now(timezone.utc) - timedelta(days=days)

# =========================
#   4) FEEDS SAMMELN (mit Limits)
# =========================
def get_new_entries(existing_keys):
    """
    Neue Einträge holen, begrenzt:
    - MAX_PER_FEED je Feed
    - MAX_TOTAL gesamt
    - nur RECENCY_DAYS alt
    - FIRST_RUN_SHALLOW: wenn keine Daten vorhanden → nur letzte MAX_PER_FEED je Feed
    """
    new_entries = []
    print("Starte das Abrufen der Feeds auf neue Einträge...")

    first_run = (len(existing_keys) == 0) and FIRST_RUN_SHALLOW

    for name, feed_info in RSS_FEEDS.items():
        if len(new_entries) >= MAX_TOTAL:
            break

        url = feed_info["url"]
        content_type = feed_info["type"]
        picked_for_feed = 0

        try:
            feed = feedparser.parse(url)
            entries = list(feed.entries)

            # Neueste zuerst (viele Feeds sind alt->neu)
            entries = entries[::-1]

            # Beim First-Run nur die letzten MAX_PER_FEED betrachten
            if first_run:
                entries = entries[:MAX_PER_FEED]

            for entry in entries:
                if len(new_entries) >= MAX_TOTAL:
                    break
                if picked_for_feed >= MAX_PER_FEED:
                    break

                uid = _make_uid(entry)
                if uid in existing_keys:
                    continue

                # Datum in UTC ISO
                if getattr(entry, "published_parsed", None):
                    published_iso = _iso_from_struct_time(entry.published_parsed)
                elif getattr(entry, "updated_parsed", None):
                    published_iso = _iso_from_struct_time(entry.updated_parsed)
                else:
                    published_iso = datetime.now(timezone.utc).isoformat()

                # Recency-Filter
                if not _is_recent(published_iso, RECENCY_DAYS):
                    continue

                content_field = entry.get("summary") or entry.get("content", [{"value": ""}])[0].get("value", "")
                clean_content = _clean_html(content_field)
                audio_url = _find_audio_url(entry) if content_type == "podcast" else ""

                new_item = {
                    "uid": uid,
                    "source": name,
                    "title": (entry.get("title") or "").strip(),
                    "link": (entry.get("link") or "").strip(),
                    "published": published_iso,
                    "content_raw": clean_content[:4000],
                    "type": content_type,
                    "audio_url": audio_url,
                }
                new_entries.append(new_item)
                picked_for_feed += 1
                print(f"-> Neu: '{new_item['title']}' [{name}]")
        except Exception as e:
            print(f"Fehler beim Abrufen von Feed {name}: {e}")

    return new_entries

# =========================
#   5) GEMINI-ANREICHERUNG
# =========================
def _coerce_result(obj: dict) -> dict:
    """Modell-Ergebnis validieren/normalisieren + Defaults setzen."""
    out = {
        "summary_ai": obj.get("summary_ai", "Zusammenfassung konnte nicht erstellt werden."),
        "topics": obj.get("topics", []),
        "sentiment": obj.get("sentiment", "neutral"),
    }
    if not isinstance(out["topics"], list):
        out["topics"] = []
    out["topics"] = [str(t) for t in out["topics"]][:3]
    while len(out["topics"]) < 3:
        out["topics"].append("")
    if out["sentiment"] not in {"positiv", "neutral", "negativ"}:
        out["sentiment"] = "neutral"
    out["summary_ai"] = str(out["summary_ai"]).strip()
    return out

def process_with_gemini(entries):
    """Mit Gemini anreichern; respektiert MAX_TOTAL und RUN_BUDGET_SEC; gibt Detail-Logs aus."""
    if not entries:
        return []

    start_ts = time.time()
    remaining = MAX_TOTAL  # Sicherheitskappung
    print(f"\nStarte die Verarbeitung von bis zu {min(len(entries), remaining)} Einträgen mit Gemini...")

    processed_entries = []
    processed_count = 0
    error_count = 0

    for entry in entries:
        if remaining <= 0:
            print("Limit erreicht (MAX_TOTAL).")
            break
        if (time.time() - start_ts) >= RUN_BUDGET_SEC:
            print(f"Zeitbudget erreicht ({RUN_BUDGET_SEC}s). Breche ab.")
            break

        try:
            full_prompt = PROMPT.format(title=entry["title"], content=entry["content_raw"])
            try:
                resp = model.generate_content(
                    full_prompt,
                    generation_config=GENERATION_CONFIG,
                    safety_settings=SAFETY_SETTINGS,
                    request_options={"timeout": 30},  # je nach SDK-Version optional/ignoriert
                )
            except TypeError:
                resp = model.generate_content(
                    full_prompt,
                    generation_config=GENERATION_CONFIG,
                    safety_settings=SAFETY_SETTINGS,
                )

            raw = (resp.text or "").strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                data = json.loads(m.group(0)) if m else {}

            coerced = _coerce_result(data)
            entry.update(coerced)
            print(f"   ✓ '{entry['title']}' → Sentiment={entry['sentiment']}, Topics={entry['topics']}")
            processed_count += 1
        except Exception as e:
            print(f"   ✗ Fehler bei '{entry.get('title','(ohne Titel)')}': {e}")
            entry.update(_coerce_result({}))
            error_count += 1
        finally:
            entry.pop("content_raw", None)
            processed_entries.append(entry)
            remaining -= 1
            time.sleep(0.2)  # leichtes Throttling

    # kurze Abschlusszeile für diesen Schritt
    print(f"Verarbeitung beendet: {processed_count} ok, {error_count} Fehler, {len(processed_entries)} Gesamt.")

    return processed_entries

# =========================
#   6) SPEICHERN
# =========================
def save_data(all_items):
    """Alle Items nach published (neueste zuerst) sortiert in DATA_FILE speichern."""
    sorted_data = sorted(all_items, key=lambda x: x.get("published", ""), reverse=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=4)
    print(f"\n✅ Daten erfolgreich in '{DATA_FILE}' gespeichert. Gesamtanzahl: {len(sorted_data)}")

# =========================
#   7) MAIN
# =========================
if __name__ == "__main__":
    start = time.time()

    existing_data = load_existing_data()
    existing_count_before = len(existing_data)

    new_entries = get_new_entries(existing_data.keys())
    fetched_count = len(new_entries)

    if new_entries:
        processed = process_with_gemini(new_entries)
        # dedupliziert über uid
        for item in processed:
            uid = item.get("uid") or item.get("link")
            if uid:
                existing_data[uid] = item
        save_data(list(existing_data.values()))

        total_after = len(existing_data)
        added = total_after - existing_count_before

        duration = int(time.time() - start)
        print(f"\n=== Zusammenfassung ===")
        print(f"Neue Items geholt:     {fetched_count}")
        print(f"Neu verarbeitet (KI):  {len(processed)}")
        print(f"Neu hinzugefügt:       {added}")
        print(f"Gesamt jetzt:          {total_after}")
        print(f"Laufzeit:              {duration}s (Budget: {RUN_BUDGET_SEC}s)")
    else:
        print("\nKeine neuen Einträge gefunden. 'data.json' wird nicht aktualisiert.")
