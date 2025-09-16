import os
import json
import feedparser
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Lade den API-Key aus der .env Datei für lokale Tests
load_dotenv()

# --- Konfiguration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

PROCESSED_LINKS_FILE = "processed_links.json"

# NEU: Die Liste enthält jetzt einen 'type' für jeden Feed
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

PROMPT = """Fasse den folgenden Inhalt in maximal zwei prägnanten Sätzen auf Deutsch für ein schnelles News-Briefing zusammen. Unabhängig von der Originalsprache, die Antwort muss auf Deutsch sein. Konzentriere dich auf das wichtigste Ergebnis oder die zentrale Nachricht. Antwort nur mit der deutschen Zusammenfassung, ohne Einleitung.
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
    except (json.JSONDecodeError, FileNotFoundError):
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
        feed_type = feed_info["type"]
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if entry.link and entry.link not in processed_links:
                    published_dt = datetime.now() # Fallback
                    if hasattr(entry, 'published_parsed') and entry.published_parsed is not None:
                        published_dt = datetime(*entry.published_parsed[:6])
                    
                    if published_dt > two_days_ago:
                        content = entry.get('summary', entry.get('content', [{'value': ''}])[0]['value'])
                        import re
                        clean_content = re.sub('<[^<]+?>', '', content)

                        new_entries.append({
                            "source": name,
                            "title": entry.title,
                            "link": entry.link,
                            "published": entry.get('published', 'N/A'),
                            "content_raw": clean_content[:2000],
                            "type": feed_type  # NEU: Der Typ wird hinzugefügt
                        })
                        print(f"-> Neuer '{feed_type}' gefunden: '{entry.title}' von '{name}'")
        except Exception as e:
            print(f"Fehler beim Abrufen von Feed {name}: {e}")
    
    return new_entries

def summarize_with_gemini(entries):
    """Erstellt Zusammenfassungen für jeden neuen Eintrag mit Gemini."""
    if not entries:
        return []
        
    print(f"\nStarte die Zusammenfassung von {len(entries)} neuen Einträgen mit Gemini...")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    for entry in entries:
        try:
            full_prompt = PROMPT.format(entry['title'] + "\n" + entry['content_raw'])
            response = model.generate_content(full_prompt)
            
            entry['summary_ai'] = response.text.strip()
            print(f"-> '{entry['title']}' erfolgreich zusammengefasst.")
        except Exception as e:
            print(f"Fehler bei der Zusammenfassung von '{entry['title']}': {e}")
            entry['summary_ai'] = "Zusammenfassung konnte nicht erstellt werden."
    
    return entries

def save_to_json(data, filename="data.json"):
    """Speichert die neuen Daten in einer JSON-Datei."""
    try:
        # Alte Daten laden, falls vorhanden
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        # Neue Daten hinzufügen und Duplikate vermeiden
        existing_links = {item['link'] for item in existing_data}
        for new_item in data:
            if new_item['link'] not in existing_links:
                existing_data.append(new_item)
        
        # Nach Datum sortieren (optional, aber schön für die App)
        # existing_data.sort(key=lambda x: feedparser.parse(x['published'])['published_parsed'], reverse=True)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Daten erfolgreich in '{filename}' gespeichert/aktualisiert.")
    except Exception as e:
        print(f"Fehler beim Speichern der JSON-Datei: {e}")

# --- Hauptablauf ---
if __name__ == "__main__":
    processed_links = load_processed_links()
    new_entries_to_process = get_new_entries(processed_links)
    
    if new_entries_to_process:
        summarized_data = summarize_with_gemini(new_entries_to_process)
        save_to_json(summarized_data)
        
        newly_processed_links = {entry['link'] for entry in new_entries_to_process}
        all_processed_links = processed_links.union(newly_processed_links)
        save_processed_links(all_processed_links)
    else:
        print("\nKeine neuen Einträge gefunden. 'data.json' wird nicht aktualisiert.")

