import os
import json
import feedparser
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# Lade den API-Key aus der .env Datei für lokale Tests
load_dotenv()

# --- Konfiguration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

PROCESSED_LINKS_FILE = "processed_links.json"

# Deine Quellenliste, jetzt mit 'type'
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


# NEU: Der verbesserte Prompt, der JSON als Antwort erzwingt
PROMPT = """Analysiere den folgenden Inhalt. Unabhängig von der Originalsprache, gib deine Antwort als valides JSON-Objekt zurück.
Das JSON-Objekt muss exakt die folgende Struktur haben:
{
  "summary_de": "Eine prägnante Zusammenfassung des Inhalts in maximal zwei deutschen Sätzen.",
  "topics": ["ein_thema", "zweites_thema", "drittes_thema"]
}
Die "topics" sollen die 3 relevantesten Schlagworte oder Konzepte des Inhalts als einzelne, kleingeschriebene Strings ohne Sonderzeichen enthalten.

Inhalt:
---
{}
---
"""

def load_processed_links():
    """Lädt die Liste der bereits verarbeiteten Links."""
    if not os.path.exists(PROCESSED_LINKS_FILE):
        return set()
    try:
        with open(PROCESSED_LINKS_FILE, "r") as f:
            return set(json.load(f))
    except json.JSONDecodeError:
        return set()

def save_processed_links(links):
    """Speichert die aktualisierte Liste der verarbeiteten Links."""
    with open(PROCESSED_LINKS_FILE, "w") as f:
        json.dump(list(links), f)

def get_new_entries(processed_links):
    """Sammelt nur die neuen Einträge aus allen RSS-Feeds."""
    new_entries = []
    # Wir schauen uns Einträge der letzten 2 Tage an, um nichts zu verpassen
    two_days_ago = datetime.now() - timedelta(days=2)
    print("Starte das Abrufen der Feeds auf neue Einträge...")

    for name, feed_info in RSS_FEEDS.items():
        url = feed_info["url"]
        entry_type = feed_info["type"]
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if entry.link not in processed_links:
                    published_time = entry.get('published_parsed')
                    if published_time:
                        published_dt = datetime(*published_time[:6])
                        if published_dt > two_days_ago:
                            content = entry.get('summary', entry.get('content', [{'value': ''}])[0]['value'])
                            import re
                            clean_content = re.sub('<[^<]+?>', '', content)

                            new_entries.append({
                                "source": name,
                                "title": entry.title,
                                "link": entry.link,
                                "published": entry.get('published', 'N/A'),
                                "content_raw": clean_content[:3000], # Etwas mehr Kontext für bessere Themen
                                "type": entry_type
                            })
                            print(f"-> Neuer Eintrag gefunden: '{entry.title}' von '{name}'")
        except Exception as e:
            print(f"Fehler beim Abrufen von Feed {name}: {e}")
    
    return new_entries

def process_with_gemini(entries):
    """Analysiert jeden Eintrag mit Gemini, um Zusammenfassung und Themen zu extrahieren."""
    if not entries:
        return []
        
    print(f"\nStarte die Analyse von {len(entries)} neuen Einträgen mit Gemini...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    generation_config = genai.types.GenerationConfig(
        response_mime_type="application/json"
    )

    processed_entries = []
    for entry in entries:
        try:
            full_prompt = PROMPT.format(entry['title'] + "\n" + entry['content_raw'])
            response = model.generate_content(full_prompt, generation_config=generation_config)
            
            # Die Antwort von Gemini ist jetzt ein JSON-String, wir müssen ihn parsen
            response_json = json.loads(response.text)
            
            entry['summary_ai'] = response_json.get('summary_de', 'Keine Zusammenfassung.')
            entry['topics'] = response_json.get('topics', [])
            
            processed_entries.append(entry)
            print(f"-> '{entry['title']}' erfolgreich analysiert.")
            
        except Exception as e:
            print(f"Fehler bei der Analyse von '{entry['title']}': {e}")
            # Füge den Eintrag trotzdem hinzu, damit der Link als verarbeitet markiert wird
            entry['summary_ai'] = "Analyse fehlgeschlagen."
            entry['topics'] = []
            processed_entries.append(entry)
        
        # Kleine Pause, um API-Limits nicht zu überschreiten
        time.sleep(1)
        
    return processed_entries


def update_data_file(new_data, filename="data.json"):
    """Lädt die alte data.json, fügt neue Daten hinzu und speichert sie."""
    
    # Entferne den rohen Inhalt, der nicht in der finalen JSON-Datei sein muss
    for item in new_data:
        del item['content_raw']

    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                # Behandelt den Fall, dass die Datei leer oder korrupt ist
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                      existing_data = []
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Füge die neuen Einträge am Anfang der Liste hinzu
        updated_data = new_data + existing_data
        
        # Behalte nur die neuesten 100 Einträge, um die Datei klein zu halten
        updated_data = updated_data[:100]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=4)
            
        print(f"\n✅ {len(new_data)} neue Einträge erfolgreich in '{filename}' hinzugefügt.")

    except Exception as e:
        print(f"Fehler beim Aktualisieren der JSON-Datei: {e}")


# --- Hauptablauf ---
if __name__ == "__main__":
    processed_links = load_processed_links()
    new_entries_to_process = get_new_entries(processed_links)
    
    if new_entries_to_process:
        analyzed_data = process_with_gemini(new_entries_to_process)
        update_data_file(analyzed_data)
        
        newly_processed_links = {entry['link'] for entry in new_entries_to_process}
        all_processed_links = processed_links.union(newly_processed_links)
        save_processed_links(all_processed_links)
    else:
        print("\nKeine neuen Einträge gefunden. 'data.json' wird nicht aktualisiert.")

