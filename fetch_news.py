import os
import json
import feedparser
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
import time

# --- Konfiguration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

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
    # Deutsche News
    "Heise (Thema KI)": {"url": "https://www.heise.de/thema/kuenstliche-intelligenz/rss.xml", "type": "article"},
    "t3n (Thema KI)": {"url": "https://t3n.de/tag/ki/rss", "type": "article"},
    "ZEIT ONLINE (Digital)": {"url": "https://newsfeed.zeit.de/digital/index", "type": "article"}
}

PROMPT = """Analysiere den folgenden Inhalt und gib eine JSON-Antwort mit zwei Schlüsseln zurück: "summary" und "topics".
- "summary": Fasse den Inhalt in maximal zwei prägnanten Sätzen auf Deutsch zusammen. Konzentriere dich auf das wichtigste Ergebnis oder die zentrale Nachricht.
- "topics": Extrahiere die 3 relevantesten Schlagworte oder Themen als eine Liste von Strings.
Der gesamte Inhalt, den du zurückgibst, MUSS valides JSON sein.

Inhalt:
---
{}
---
"""

def get_existing_data():
    """Lädt bestehende Daten aus data.json, falls vorhanden."""
    if not os.path.exists(DATA_FILE):
        return [], set()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            existing_articles = json.load(f)
            existing_links = {article['link'] for article in existing_articles}
            return existing_articles, existing_links
    except (json.JSONDecodeError, FileNotFoundError):
        return [], set()

def get_new_entries(existing_links):
    """Sammelt nur die neuen, noch nicht verarbeiteten Einträge."""
    new_entries = []
    print("Starte das Abrufen der Feeds auf neue Einträge...")

    for name, feed_info in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_info['url'])
            for entry in feed.entries:
                if entry.link and entry.link not in existing_links:
                    content = entry.get('summary', entry.get('content', [{'value': ''}])[0]['value'])
                    import re
                    clean_content = re.sub('<[^<]+?>', '', content)
                    
                    published_parsed = entry.get('published_parsed')
                    if published_parsed:
                        published_dt = datetime.fromtimestamp(time.mktime(published_parsed))
                        published_iso = published_dt.isoformat()
                    else:
                        published_iso = datetime.now().isoformat()

                    # NEU: Suche nach der direkten Audio-URL für Podcasts
                    audio_url = ""
                    if feed_info['type'] == 'podcast' and 'enclosures' in entry:
                        for enclosure in entry.enclosures:
                            if 'audio' in enclosure.get('type', ''):
                                audio_url = enclosure.href
                                break
                    
                    new_entries.append({
                        "source": name,
                        "type": feed_info['type'],
                        "title": entry.title,
                        "link": entry.link,
                        "published": published_iso,
                        "content_raw": clean_content[:4000],
                        "audio_url": audio_url  # Füge die gefundene URL hinzu
                    })
                    existing_links.add(entry.link)
        except Exception as e:
            print(f"Fehler beim Abrufen von Feed {name}: {e}")
    
    print(f"{len(new_entries)} neue Einträge gefunden.")
    return new_entries

def process_with_gemini(entries):
    """Verarbeitet neue Einträge mit der Gemini API."""
    if not entries:
        return []
        
    print(f"\nStarte die Verarbeitung von {len(entries)} neuen Einträgen mit Gemini...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config={"response_mime_type": "application/json"})
    processed_articles = []

    for entry in entries:
        try:
            full_prompt = PROMPT.format(entry['title'] + "\n" + entry['content_raw'])
            response = model.generate_content(full_prompt)
            response_json = json.loads(response.text)
            
            processed_article = {
                "source": entry["source"],
                "title": entry["title"],
                "link": entry["link"],
                "published": entry["published"],
                "type": entry["type"],
                "audio_url": entry["audio_url"], # Stelle sicher, dass die URL weitergegeben wird
                "summary_ai": response_json.get("summary", "Zusammenfassung konnte nicht erstellt werden."),
                "topics": response_json.get("topics", [])
            }
            processed_articles.append(processed_article)
            print(f"-> '{entry['title']}' erfolgreich verarbeitet.")
            time.sleep(1)
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von '{entry['title']}': {e}")
    
    return processed_articles

def save_data(all_articles):
    """Speichert die kombinierte und sortierte Liste in data.json."""
    all_articles.sort(key=lambda x: x['published'], reverse=True)
    
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Daten erfolgreich aktualisiert. Gesamtanzahl der Einträge: {len(all_articles)}.")
    except Exception as e:
        print(f"Fehler beim Speichern der JSON-Datei: {e}")

# --- Hauptablauf ---
if __name__ == "__main__":
    existing_articles, existing_links = get_existing_data()
    new_entries_to_process = get_new_entries(existing_links)
    
    if new_entries_to_process:
        newly_processed_articles = process_with_gemini(new_entries_to_process)
        combined_articles = existing_articles + newly_processed_articles
        save_data(combined_articles)
    else:
        print("\nKeine neuen Einträge gefunden. 'data.json' wird nicht aktualisiert.")

