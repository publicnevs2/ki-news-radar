import os
import json
import feedparser
from datetime import datetime, timezone
import re
import time # Wichtig für die Höflichkeitspause
import google.generativeai as genai
from dotenv import load_dotenv

# Lade den API-Key aus der .env Datei für lokale Tests
load_dotenv()

# --- Konfiguration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

DATA_FILE = "data.json"
MAX_ENTRIES_PER_RUN = 20 # Verarbeite nur 20 Einträge pro Lauf, um Timeouts und Limits zu vermeiden

RSS_FEEDS = {
    # Deutsche Podcasts
    "Heise KI Update": {"url": "https://kiupdate.podigee.io/feed/mp3", "type": "podcast"},
    "AI First": {"url": "https://feeds.captivate.fm/ai-first/", "type": "podcast"},
    "KI Inside": {"url": "https://agidomedia.podcaster.de/insideki.rss", "type": "podcast"},
    "KI>Inside": {"url": "https://anchor.fm/s/fb4ad23c/podcast/rss", "type": "podcast"},
    "Your CoPilot": {"url": "https://podcast.yourcopilot.de/feed/mp3", "type": "podcast"},
    # Deutsche News
    "Heise (Thema KI)": {"url": "https://www.heise.de/thema/kuenstliche-intelligenz/rss.xml", "type": "article"},
    "t3n (Thema KI)": {"url": "https://t3n.de/tag/ki/rss", "type": "article"},
}

PROMPT = """Analysiere den folgenden Inhalt. Gib deine Antwort ausschließlich als valides JSON-Objekt zurück, ohne umschließende Markdown-Syntax. Das JSON-Objekt soll folgende Struktur haben:
{
  "summary": "Eine prägnante Zusammenfassung des Inhalts in maximal zwei Sätzen auf Deutsch. Unabhängig von der Originalsprache, die Antwort muss auf Deutsch sein.",
  "topics": ["ein-Haupt-Thema", "ein-weiteres-Thema", "ein-drittes-Thema"]
}
Inhalt:
---
{}
---
"""

def load_json_file(filename, default_value):
    """Lädt eine JSON-Datei sicher."""
    if not os.path.exists(filename): return default_value
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError): return default_value

def save_json_file(data, filename):
    """Speichert Daten sicher in einer JSON-Datei."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e: print(f"Fehler beim Speichern von {filename}: {e}")

def get_new_entries(processed_links):
    """Sammelt neue Einträge aus den Feeds, sortiert sie und begrenzt die Anzahl."""
    all_new_entries = []
    print("Starte das Abrufen der Feeds auf neue Einträge...")
    for name, feed_info in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries:
                if entry.link not in processed_links:
                    published_time = entry.get('published_parsed')
                    dt_object = datetime.now(timezone.utc)
                    if published_time:
                        dt_object = datetime(*published_time[:6], tzinfo=timezone.utc)

                    content = entry.get('summary', entry.get('content', [{'value': ''}])[0]['value'])
                    clean_content = re.sub('<[^<]+?>', '', content)
                    audio_url = ""
                    if feed_info["type"] == "podcast" and "enclosures" in entry:
                         for enc in entry.enclosures:
                            if "audio" in enc.get("type", ""): audio_url = enc.href; break
                    
                    all_new_entries.append({
                        "source": name, "title": entry.title, "link": entry.link,
                        "published": dt_object.isoformat(), "type": feed_info["type"],
                        "content_raw": clean_content[:2000], "audio_url": audio_url
                    })
        except Exception as e: print(f"Fehler beim Abrufen von Feed {name}: {e}")
    
    all_new_entries.sort(key=lambda x: x['published'], reverse=True)
    limited_entries = all_new_entries[:MAX_ENTRIES_PER_RUN]
    print(f"{len(all_new_entries)} neue Einträge gefunden. Verarbeite die {len(limited_entries)} neuesten.")
    return limited_entries

def process_with_gemini(entries):
    """Verarbeitet neue Einträge mit Gemini, inklusive Pausen und robuster JSON-Extraktion."""
    if not entries: return []
    
    print(f"\nStarte die Verarbeitung von {len(entries)} neuen Einträgen mit Gemini...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config={"response_mime_type": "application/json"})
    
    processed_entries = []
    for entry in entries:
        try:
            full_prompt = PROMPT.format(entry['title'] + "\n" + entry['content_raw'])
            response = model.generate_content(full_prompt)
            
            # KORREKTUR: Robuste JSON-Extraktion statt blindem Parsen
            response_text = response.text.strip()
            # Finde den ersten '{' und den letzten '}' um sicherzustellen, dass wir nur das JSON-Objekt haben
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            
            if start_index != -1 and end_index != -1:
                json_str = response_text[start_index:end_index+1]
                json_data = json.loads(json_str)
                entry['summary_ai'] = json_data.get('summary', 'Keine Zusammenfassung.')
                entry['topics'] = json_data.get('topics', [])
                del entry['content_raw']
                processed_entries.append(entry)
                print(f"-> '{entry['title']}' erfolgreich verarbeitet.")
            else:
                print(f"FEHLER: Kein valides JSON in der Gemini-Antwort für '{entry['title']}' gefunden.")

        except Exception as e:
            print(f"FEHLER bei der Verarbeitung von '{entry['title']}': {e}")
        
        finally:
            time.sleep(1) # Höflichkeitspause
            
    return processed_entries

# --- Hauptablauf ---
if __name__ == "__main__":
    existing_data = load_json_file(DATA_FILE, [])
    processed_links = set(item['link'] for item in existing_data)
    
    new_entries = get_new_entries(processed_links)
    
    if new_entries:
        successfully_processed = process_with_gemini(new_entries)
        
        if successfully_processed:
            combined_data = existing_data + successfully_processed
            combined_data.sort(key=lambda x: x['published'], reverse=True)
            
            save_json_file(combined_data, DATA_FILE)
            print(f"\n✅ {len(successfully_processed)} neue Einträge hinzugefügt. Gesamt: {len(combined_data)}.")
        else:
            print("\nKeine neuen Einträge konnten erfolgreich verarbeitet werden.")
    else:
        print("\nKeine neuen Einträge in den Feeds gefunden.")

