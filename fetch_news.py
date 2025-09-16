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
from google.generativeai import types  # für GenerationConfig & Safety-Enums

# ------------------------------------------------------------
# 1) ENV laden & Gemini konfigurieren
# ------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("FEHLER: GEMINI_API_KEY nicht gefunden. Skript wird beendet.")
    raise SystemExit(1)

genai.configure(api_key=GEMINI_API_KEY)

generation_config = types.GenerationConfig(
    response_mime_type="application/json",  # wichtig: 'application/json'
)

safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

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
    "KDnuggets": {"url": "https://www.kdnuggets.com/feed", "type": "article"},
    "OpenAI Blog": {"url": "https://openai.com/blog/rss.xml", "type": "article"},
    "Google AI Blog": {"url": "https://ai.googleblog.com/feeds/posts/default?alt=rss", "type": "article"},
    "Hugging Face Blog": {"url": "https://huggingface.co/blog/feed.xml", "type": "article"},
    "DeepMind Blog": {"url": "https://deepmind.google/discover/blog/feed/basic", "type": "article"},
    "BAIR Blog (Berkeley AI Research)": {"url": "https://bair.berkeley.edu/blog/feed.xml", "type": "article"},
    "AI Trends": {"url": "https://aitrends.com/feed/", "type": "article"},
    "NVIDIA Blog (AI)": {"url": "https://blogs.nvidia.com/blog/category/ai/feed/", "type": "article"},
    "VentureBeat AI": {"url": "https://venturebeat.com/category/ai/feed/", "type": "article"},

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
    Lädt die bestehende Daten-Datei, falls vorhanden, und bildet ein Dict
    key -> item. Als Key wird 'uid' (falls vorhanden) oder Link genutzt.
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
    """Entfernt HTML-Tags, unescaped Entities und normalisiert Whitespace."""
    if not text:
        return ""
    no_tags = re.sub(r"<[^>]+?>", "", text)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def _iso_from_struct_time(st) -> str:
    """Konvertiert feedparser struct_time nach UTC ISO 8601."""
    if not st:
        return datetime.now(timezone.utc).isoformat()
    dt = datetime.fromtimestamp(time.mktime(st), tz=timezone.utc)
    return dt.isoformat()


def _find_audio_url(entry) -> str:
    """
    Sucht eine Audio-URL in Enclosures (feedparser) oder Links mit rel='enclosure'.
    """
    # 1) Klassische enclosures
    try:
        if hasattr(entry, "enclosures"):
            for enc in entry.enclosures:
                # enc kann ein dict oder ein Objekt sein
                t = enc.get("type", "") if isinstance(enc, dict) else getattr(enc, "type", "")
                href = enc.get("href", "") if isinstance(enc, dict) else getattr(enc, "href", "")
                if "audio" in (t or "") and href:
                    return href
    except Exception:
        pass

    # 2) Links mit rel='enclosure'
    try:
        for l in getattr(entry, "links", []):
            if l.get("rel") == "enclosure" and "audio" in l.get("type", ""):
                return l.get("href", "")
    except Exception:
        pass

    return ""


def _make_uid(entry) -> str:
    """Erstellt eine stabile UID aus entry.id (falls verfügbar) sonst entry.link."""
    uid = getattr(entry, "id", None) or getattr(entry, "guid", None) or getattr(entry, "link", None)
    return uid or f"no-id::{getattr(entry, 'title', '')[:50]}::{time.time()}"


def get_new_entries(existing_keys):
    """
    Sammelt neue Einträge aus allen RSS-Feeds und gibt eine Liste von Roh-Einträgen zurück,
    die anschließend via Gemini angereichert werden.
    """
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

                # Inhalt extrahieren
                content_field = entry.get("summary")
                if not content_field:
                    content_field = entry.get("content", [{"value": ""}])[0].get("value", "")
                clean_content = _clean_html(content_field)

                # Audio ermitteln
                audio_url = _find_audio_url(entry) if content_type == "podcast" else ""

                # Datum in UTC ISO
                if getattr(entry, "published_parsed", None):
                    published_iso = _iso_from_struct_time(entry.published_parsed)
                elif getattr(entry, "updated_parsed", None):
                    published_iso = _iso_from_struct_time(entry.updated_parsed)
                else:
                    published_iso = datetime.now(timezone.utc).isoformat()

                new_item = {
                    "uid": uid,  # stabiler Schlüssel für Deduplizierung
                    "source": name,
                    "title": entry.get("title", "").strip(),
                    "link": entry.get("link", "").strip(),
                    "published": published_iso,
                    "content_raw": clean_content[:4000],  # Prompt-Context begrenzen
                    "type": content_type,
                    "audio_url": audio_url,
                }
                new_entries.append(new_item)
                print(f"-> Neuer Eintrag gefunden: '{new_item['title']}' von '{name}'")
        except Exception as e:
            print(f"Fehler beim Abrufen von Feed {name}: {e}")

    return new_entries


def _coerce_result(obj: dict) -> dict:
    """Validiert/normalisiert das Modell-Ergebnis und füllt Defaults."""
    out = {
        "summary_ai": obj.get("summary_ai", "Zusammenfassung konnte nicht erstellt werden."),
        "topics": obj.get("topics", []),
        "sentiment": obj.get("sentiment", "neutral"),
    }

    # Topics: Liste, genau 3 Elemente (auffüllen/trimmen)
    if not isinstance(out["topics"], list):
        out["topics"] = []
    out["topics"] = [str(t) for t in out["topics"]][:3]
    while len(out["topics"]) < 3:
        out["topics"].append("")

    # Sentiment: nur erlaubte Werte
    if out["sentiment"] not in {"positiv", "neutral", "negativ"}:
        out["sentiment"] = "neutral"

    # Summary trimmen
    out["summary_ai"] = str(out["summary_ai"]).strip()
    return out


def process_with_gemini(entries):
    """
    Reicht Einträge an das Gemini-Modell weiter und fügt
    summary_ai, topics, sentiment hinzu. Fällt robust auf Defaults zurück.
    """
    if not entries:
        return []

    print(f"\nStarte die Verarbeitung von {len(entries)} neuen Einträgen mit Gemini...")
    processed_entries = []

    for entry in entries:
        try:
            full_prompt = PROMPT.format(title=entry["title"], content=entry["content_raw"])
            resp = model.generate_content(full_prompt)
            raw = (resp.text or "").strip()

            # Hartes JSON-Parsing (falls Modell doch Text um das JSON legt)
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
            time.sleep(0.6)  # sanfter bzgl. Rate-Limits

    return processed_entries


def save_data(all_items):
    """
    Speichert alle Items in DATA_FILE, sortiert nach published (neueste zuerst).
    """
    # Sicherheit: published ist ISO-String; Sortierung lexikografisch funktioniert für UTC-ISO
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

        # In bestehendes Dict einpflegen (dedupliziert über uid)
        for item in processed:
            uid = item.get("uid") or item.get("link")
            if uid:
                existing_data[uid] = item

        # Persistieren
        save_data(list(existing_data.values()))
    else:
        print("\nKeine neuen Einträge gefunden. 'data.json' wird nicht aktualisiert.")
