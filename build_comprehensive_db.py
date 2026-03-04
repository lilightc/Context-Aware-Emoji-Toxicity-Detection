import json
import os
import emoji
import requests
import time
import urllib.parse
from bs4 import BeautifulSoup
from datasets import load_dataset
from collections import Counter

# --- CONFIGURATION ---
OUTPUT_FILE = "data/slang_dictionary_complete.json"
ACADEMIC_SOURCE = "HannahRoseKirk/HatemojiBuild"

def get_urban_definition(term):
    """Scrapes the top definition from Urban Dictionary safely."""
    safe_term = urllib.parse.quote(term)
    url = f"https://www.urbandictionary.com/define.php?term={safe_term}"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/91.0.4472.114"}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200: return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        def_block = soup.find('div', class_='definition')
        
        if def_block:
            return def_block.get_text().replace('[', '').replace(']', '')
    except:
        return None
    return None

def build_comprehensive_pipeline():
    print("🚀 Starting Comprehensive Data Pipeline (No Limits)...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 1. Load Academic Data
    print("📥 Loading Hatemoji dataset...")
    hf_token = os.environ["HF_TOKEN"]
    try:
        dataset = load_dataset(ACADEMIC_SOURCE, token=hf_token)
    except Exception as exc:
        if "gated dataset" in str(exc).lower():
            msg = (
                "HatemojiBuild is gated on the Hub. Set HF_TOKEN with access, e.g.\n"
                "  export HF_TOKEN=your_token\n"
                "or login via `huggingface-cli login` and retry."
            )
            raise RuntimeError(msg) from exc
        raise
    
    # 2. Extract ALL unique emojis from the toxic subset
    toxic_emojis = set()
    print("🔍 Scanning for emojis in toxic comments...")
    
    for row in dataset['train']:
        if row['label_gold'] == 1: # Only look at confirmed hate speech
            # Extract every emoji character found in the text
            found = [c for c in row['text'] if c in emoji.EMOJI_DATA]
            toxic_emojis.update(found)
            
    print(f"   -> Found {len(toxic_emojis)} unique emojis associated with toxicity.")
    
    # 3. Scrape Definitions for ALL of them
    print(f"🕷️ Beginning scraping for {len(toxic_emojis)} emojis.")
    print("   (This will take a few minutes to avoid IP bans...)")
    
    final_database = []
    
    # Load existing progress if script crashed previously
    try:
        with open(OUTPUT_FILE, 'r') as f:
            final_database = json.load(f)
            scraped_ids = {entry['symbol'] for entry in final_database}
            print(f"   -> Resuming... {len(final_database)} already scraped.")
    except FileNotFoundError:
        scraped_ids = set()

    # Convert set to list for iteration
    target_list = list(toxic_emojis)
    
    for i, symbol in enumerate(target_list):
        if symbol in scraped_ids:
            continue # Skip if already done
            
        print(f"   [{i+1}/{len(target_list)}] Scraping {symbol} ...", end=" ")
        
        # A. Try Urban Dictionary
        definition = get_urban_definition(symbol)
        
        # B. Construct Entry
        emoji_id = "emoji_" + "-".join(f"{ord(ch):x}" for ch in symbol)
        entry = {
            "id": emoji_id,  # Unique Unicode ID (supports multi-codepoint emojis)
            "symbol": symbol,
            "slang_definition": definition if definition else "No slang definition found.",
            "source": "Urban Dictionary" if definition else "Academic Dataset Only",
            "is_tracked_in_academic": True
        }
        
        final_database.append(entry)
        
        if definition:
            print("✅ Found slang.")
        else:
            print("❌ No slang found.")
            
        # C. SAVE PROGRESS (Critical step!)
        # We save after every 10 items so you don't lose data
        if i % 10 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(final_database, f, indent=2, ensure_ascii=False)
        
        # D. RATE LIMIT PROTECTION
        # Sleep 1.5 seconds between requests. 
        # 150 emojis * 1.5s = ~4 minutes total run time.
        time.sleep(1.5)

    # Final Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_database, f, indent=2, ensure_ascii=False)
        
    print(f"\n🎉 DONE. Comprehensive database saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_comprehensive_pipeline()