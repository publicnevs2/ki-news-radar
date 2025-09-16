import os
import json
import feedparser
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import time

# Lade den API-Key aus der .env Datei für lokale Tests
load_dotenv()

# --- Konfiguration ---
# Hol den API-Key (entweder lokal oder aus den GitHub Secrets)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Konfiguriere das Gemini-Modell mit Sicherheitseinstellungen
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = genai.GenerationConfig(
        response_mime_type="json",
    )
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    model = genai.GenerativeModel(
        'gemini-1.5-flash-latest',
        generation_config=generation_config,
        safety_settings=safety_settings
    )
else:
    print("FEHLER: GEMINI_API_KEY nicht gefunden. Skript wird beendet.")
    exit()

# Dateien für die Speicherung
DATA_FILE = "data.json"

# Deine Feed-Liste mit Typ-Information
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
    # Deutsche News
    "Heise (Thema KI)": {"url": "https://www.heise.de/thema/kuenstliche-intelligenz/rss.xml", "type": "article"},
    "t3n (Thema KI)": {"url": "https://t3n.de/tag/ki/rss", "type": "article"},
    "ZEIT ONLINE (Digital)": {"url": "https://newsfeed.zeit.de/digital/index", "type": "article"}
}

# Der Prompt für Gemini, der eine JSON-Antwort erzwingt
PROMPT = """
Analysiere den folgenden Inhalt (Titel und Textauszug). Unabhängig von der Originalsprache, gib deine Antwort als JSON-Objekt auf Deutsch zurück.
Das JSON-Objekt muss exakt die folgenden drei Schlüssel haben:
1. "summary_ai": Eine prägnante Zusammenfassung in maximal zwei Sätzen.
2. "topics": Ein Array mit den drei relevantesten Schlagwörtern/Themen als Strings.
3. "sentiment": Eine Einschätzung der Grundstimmung des Inhalts als einer dieser drei Werte: "positiv", "negativ", oder "neutral".

Inhalt:
---
Titel: {title}
Text: {content}
---
"""

def load_existing_data():
    """Lädt die bestehende Daten-Datei, falls vorhanden."""
    if not os.path.exists(DATA_FILE):
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item['link']: item for item in data}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def get_new_entries(existing_links):
    """Sammelt nur die neuen Einträge aus allen RSS-Feeds."""
    new_entries = []
    print("Starte das Abrufen der Feeds auf neue Einträge...")

    for name, feed_info in RSS_FEEDS.items():
        url = feed_info["url"]
        content_type = feed_info["type"]
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if entry.link not in existing_links:
                    content = entry.get('summary', entry.get('content', [{'value': ''}])[0]['value'])
                    import re
                    clean_content = re.sub('<[^<]+?>', '', content)
                    
                    audio_url = ""
                    if content_type == 'podcast' and 'enclosures' in entry:
                        for enc in entry.enclosures:
                            if 'audio' in enc.get('type', ''):
                                audio_url = enc.href
                                break
                    
                    # Parsen des Datums und Umwandlung in ISO 8601 Format
                    published_iso = datetime.now().isoformat()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed is not None:
                        published_dt = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                        published_iso = published_dt.isoformat()
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed is not None:
                        published_dt = datetime.fromtimestamp(time.mktime(entry.updated_parsed))
                        published_iso = published_dt.isoformat()


                    new_entries.append({
                        "source": name,
                        "title": entry.title,
                        "link": entry.link,
                        "published": published_iso,
                        "content_raw": clean_content[:4000],
                        "type": content_type,
                        "audio_url": audio_url
                    })
                    print(f"-> Neuer Eintrag gefunden: '{entry.title}' von '{name}'")
        except Exception as e:
            print(f"Fehler beim Abrufen von Feed {name}: {e}")
    
    return new_entries

def process_with_gemini(entries):
    """Reichert Einträge mit KI-generierten Daten an."""
    if not entries:
        return []
        
    print(f"\nStarte die Verarbeitung von {len(entries)} neuen Einträgen mit Gemini...")
    processed_entries = []

    for entry in entries:
        try:
            full_prompt = PROMPT.format(title=entry['title'], content=entry['content_raw'])
            response = model.generate_content(full_prompt)
            
            # Die Antwort von Gemini ist ein String, der JSON enthält -> parsen
            response_json = json.loads(response.text)
            
            # Füge die KI-Daten zum Eintrag hinzu
            entry.update(response_json)
            del entry['content_raw']  # Roh-Inhalt nach Verarbeitung entfernen
            processed_entries.append(entry)

            print(f"-> '{entry['title']}' erfolgreich verarbeitet.")
            time.sleep(1) # Kurze Pause, um API-Limits zu respektieren

        except Exception as e:
            print(f"Fehler bei der Verarbeitung von '{entry['title']}': {e}")
            # Füge Standardwerte hinzu, damit die App nicht abstürzt
            entry['summary_ai'] = "Zusammenfassung konnte nicht erstellt werden."
            entry['topics'] = []
            entry['sentiment'] = "unbekannt"
            del entry['content_raw']
            processed_entries.append(entry)
    
    return processed_entries

def save_data(all_data):
    """Speichert die kombinierten und sortierten Daten in einer JSON-Datei."""
    # Sortiere alle Einträge nach Datum, die neuesten zuerst
    sorted_data = sorted(all_data, key=lambda x: x['published'], reverse=True)
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted_data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Daten erfolgreich in '{DATA_FILE}' gespeichert. Gesamtanzahl: {len(sorted_data)}")
    except Exception as e:
        print(f"Fehler beim Speichern der JSON-Datei: {e}")

# --- Hauptablauf ---
if __name__ == "__main__":
    existing_data_dict = load_existing_data()
    new_entries_to_process = get_new_entries(existing_data_dict.keys())
    
    if new_entries_to_process:
        processed_new_entries = process_with_gemini(new_entries_to_process)
        
        # Füge die neuen, verarbeiteten Einträge zum existierenden Dictionary hinzu
        for entry in processed_new_entries:
            existing_data_dict[entry['link']] = entry
        
        # Speichere die kombinierte Liste aller Einträge
        save_data(list(existing_data_dict.values()))
    else:
        print("\nKeine neuen Einträge gefunden. 'data.json' wird nicht aktualisiert.")

