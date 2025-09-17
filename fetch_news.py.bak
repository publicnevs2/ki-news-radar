#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import html
import time
import feedparser
from dotenv import load_dotenv
from datetime import datetime, timezone

import google.generativeai as genai

# ------------------------------------------------------------
# 1) ENV laden & Gemini konfigurieren (versionsrobust)
# ------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("FEHLER: GEMINI_API_KEY nicht gefunden. Skript wird beendet.")
    raise SystemExit(1)

genai.configure(api_key=GEMINI_API_KEY)

GENERATION_CONFIG = {
    "response_mime_type": "application/json",  # wichtig: 'application/json'
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel("gemini-1.5-flash-latest")

# ------------------------------------------------------------
# 2) Konfiguration & Feeds
# ------------------------------------------------------------
DATA_FILE = "data.json"

RSS_FEEDS = {
    # Deutsche Podcasts
    "Heise KI Update": {"url": "https://kiupdate.podigee.io/feed/mp3", "type": "podcast"},
    "AI First": {"url": "https://feeds.captivate.fm/ai-first/", "type": "podcast"},
    "KI Inside": {"url": "https://agidomedia.podcaster.de/insideki.rss", "type": "podcast"},
    "KI>Inside": {"url": "https://anchor.fm/s/fb4ad23c/podcast/rss", "type": "podcast"},
    "Your CoPilot": {"url": "https://podcast.yourcopilot.de/feed/mp3", "type": "podcast"},
    "KI verstehen": {"url": "https://www.deutschlandfunk.de/ki-verstehen-102.xml", "type": "podcast"},

    # Englische News & Blogs
    "MIT Technology Review": {"url": "https://www.technologyreview.com/feed/", "type": "article"},
   # "OpenAI Blog": {"url": "https://openai.com/blog/rss.xml", "type": "article"},
   # "Google AI Blog": {"url": "https://ai.googleblog.com/feeds/posts/default?alt=rss", "type": "article"},
    "AI Trends": {"url": "https://aitrends.com/feed/", "type": "article"},
   # "NVIDIA Blog (AI)": {"url": "https://blogs.nvidia.com/blog/category/ai/feed/", "type": "article"},
    
    # Deutsche News
    "Heise (Thema KI)": {"url": "https://www.heise.de/thema/kuenstliche-intelligenz/rss.xml", "type": "article"},
    "t3n (Thema KI)": {"url": "https://t3n.de/tag/ki/rss", "type": "article"},
    "ZEIT ONLINE (Digital)": {"url": "https://newsfeed.zeit.de/digital/index", "type": "article"},
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

# ------------------------------------------------------------
# 3) Hilfsfunktionen
# ------------------------------------------------------------
def load_existing_data():
    """
    Lädt bestehende Daten-Datei und bildet ein Dict: uid -> item.
    """
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
    """feedparser struct_time -> UTC ISO 8601."""
    if not st:
        return datetime.now(timezone.utc).isoformat()
    dt = datetime.fromtimestamp(time.mktime(st), tz=timezone.utc)
    return dt.isoformat()


def _find_audio_url(entry) -> str:
    """Audio-URL aus Enclosures oder Links mit rel='enclosure' extrahieren."""
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
    """UID aus id/guid/link bilden."""
    uid = getattr(entry, "id", None) or getattr(entry, "guid", None) or getattr(entry, "link", None)
    return uid or f"no-id::{getattr(entry, 'title', '')[:50]}::{time.time()}"


def get_new_entries(existing_keys):
    """Neue Einträge aus allen RSS-Feeds sammeln (roh, noch ohne KI-Anreicherung)."""
    new_entries = []
    print("Starte das Abrufen der Feeds auf neue Einträge...")

    for name, feed_info in RSS_FEEDS.items():
        url = feed_info["url"]
        content_type = feed_info["type"]
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                uid = _make_uid(entry)
                if uid in existing_keys:
                    continue

                content_field = entry.get("summary")
                if not content_field:
                    content_field = entry.get("content", [{"value": ""}])[0].get("value", "")
                clean_content = _clean_html(content_field)

                audio_url = _find_audio_url(entry) if content_type == "podcast" else ""

                if getattr(entry, "published_parsed", None):
                    published_iso = _iso_from_struct_time(entry.published_parsed)
                elif getattr(entry, "updated_parsed", None):
                    published_iso = _iso_from_struct_time(entry.updated_parsed)
                else:
                    published_iso = datetime.now(timezone.utc).isoformat()

                new_item = {
                    "uid": uid,
                    "source": name,
                    "title": entry.get("title", "").strip(),
                    "link": entry.get("link", "").strip(),
                    "published": published_iso,
                    "content_raw": clean_content[:4000],
                    "type": content_type,
                    "audio_url": audio_url,
                }
                new_entries.append(new_item)
                print(f"-> Neuer Eintrag gefunden: '{new_item['title']}' von '{name}'")
        except Exception as e:
            print(f"Fehler beim Abrufen von Feed {name}: {e}")

    return new_entries


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
    """Mit Gemini anreichern: summary_ai, topics, sentiment (robust, JSON-only)."""
    if not entries:
        return []

    print(f"\nStarte die Verarbeitung von {len(entries)} neuen Einträgen mit Gemini...")
    processed_entries = []

    for entry in entries:
        try:
            full_prompt = PROMPT.format(title=entry["title"], content=entry["content_raw"])
            resp = model.generate_content(
                full_prompt,
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS,
            )
            raw = (resp.text or "").strip()

            # JSON-Parsing (falls das Modell doch Text drumherum liefert)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                data = json.loads(m.group(0)) if m else {}

            coerced = _coerce_result(data)
            entry.update(coerced)
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von '{entry.get('title','(ohne Titel)')}': {e}")
            entry.update(_coerce_result({}))
        finally:
            entry.pop("content_raw", None)
            processed_entries.append(entry)
            time.sleep(0.6)  # Rate-Limits schonen

    return processed_entries


def save_data(all_items):
    """Alle Items nach published (neueste zuerst) sortiert in DATA_FILE speichern."""
    sorted_data = sorted(all_items, key=lambda x: x.get("published", ""), reverse=True)
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Daten erfolgreich in '{DATA_FILE}' gespeichert. Gesamtanzahl: {len(sorted_data)}")
    except Exception as e:
        print(f"Fehler beim Speichern der JSON-Datei: {e}")


# ------------------------------------------------------------
# 4) Hauptprogramm
# ------------------------------------------------------------
if __name__ == "__main__":
    existing_data = load_existing_data()
    new_entries = get_new_entries(existing_data.keys())

    if new_entries:
        processed = process_with_gemini(new_entries)
        for item in processed:
            uid = item.get("uid") or item.get("link")
            if uid:
                existing_data[uid] = item
        save_data(list(existing_data.values()))
    else:
        print("\nKeine neuen Einträge gefunden. 'data.json' wird nicht aktualisiert.")
