import os
import json
import feedparser
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Lade den API-Key aus der .env Datei für lokale Tests
load_dotenv()

# --- Konfiguration ---
# Hol den API-Key (entweder lokal oder aus den GitHub Secrets)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Datei, um bereits verarbeitete Links zu speichern
PROCESSED_LINKS_FILE = "processed_links.json"

# Deine neue, umfangreiche Liste an RSS-Feeds
RSS_FEEDS = {
    # Deutsche Podcasts
    "Heise KI Update": "https://kiupdate.podigee.io/feed/mp3",
    "AI First": "https://feeds.captivate.fm/ai-first/",
    "KI Inside": "https://agidomedia.podcaster.de/insideki.rss",
    "KI>Inside": "https://anchor.fm/s/fb4ad23c/podcast/rss",
    "Your CoPilot": "https://podcast.yourcopilot.de/feed/mp3",
    "KI verstehen": "https://www.deutschlandfunk.de/ki-verstehen-102.xml",
    # Englische News & Blogs
    "MIT Technology Review": "https://www.technologyreview.com/feed/",
    "KDnuggets": "https://www.kdnuggets.com/feed",
    "OpenAI Blog": "https://openai.com/blog/rss.xml",
    "Google AI Blog": "https://ai.googleblog.com/feeds/posts/default?alt=rss",
    "Hugging Face Blog": "https://huggingface.co/blog/feed.xml",
    # Deutsche News
    "Heise (Thema KI)": "https://www.heise.de/thema/kuenstliche-intelligenz/rss.xml",
    "t3n (Thema KI)": "https://t3n.de/tag/ki/rss",
    "ZEIT ONLINE (Digital)": "https://newsfeed.zeit.de/digital/index"
}

# Der angepasste Befehl für die KI (erzwingt deutsche Antwort)
PROMPT = """Fasse den folgenden Inhalt in maximal zwei prägnanten Sätzen auf Deutsch für ein schnelles News-Briefing zusammen. Unabhängig von der Originalsprache, die Antwort muss auf Deutsch sein. Konzentriere dich auf das wichtigste Ergebnis oder die zentrale Nachricht. Antworte nur mit der deutschen Zusammenfassung, ohne Einleitung.
Inhalt:
---
{}
---
"""

def load_processed_links():
    """Lädt die Liste der bereits verarbeiteten Links."""
    if not os.path.exists(PROCESSED_LINKS_FILE):
        return set()
    with open(PROCESSED_LINKS_FILE, "r") as f:
        return set(json.load(f))

def save_processed_links(links):
    """Speichert die aktualisierte Liste der verarbeiteten Links."""
    with open(PROCESSED_LINKS_FILE, "w") as f:
        json.dump(list(links), f)

def get_new_entries(processed_links):
    """Sammelt nur die neuen Einträge aus allen RSS-Feeds."""
    new_entries = []
    two_days_ago = datetime.now() - timedelta(days=2)
    print("Starte das Abrufen der Feeds auf neue Einträge...")

    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # Prüfe, ob der Link neu ist
                if entry.link not in processed_links:
                    # Prüfe, ob der Eintrag aktuell ist (max 2 Tage alt)
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
                                 "content_raw": clean_content[:2000] # Begrenze die Länge
                             })
                             print(f"-> Neuer Eintrag gefunden: '{entry.title}' von '{name}'")
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
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n✅ {len(data)} neue Einträge erfolgreich in '{filename}' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der JSON-Datei: {e}")

# --- Hauptablauf ---
if __name__ == "__main__":
    processed_links = load_processed_links()
    new_entries_to_process = get_new_entries(processed_links)
    
    if new_entries_to_process:
        summarized_data = summarize_with_gemini(new_entries_to_process)
        save_to_json(summarized_data)
        
        # Aktualisiere die Liste der verarbeiteten Links
        newly_processed_links = {entry['link'] for entry in new_entries_to_process}
        all_processed_links = processed_links.union(newly_processed_links)
        save_processed_links(all_processed_links)
    else:
        print("\nKeine neuen Einträge gefunden. 'data.json' wird nicht aktualisiert.")