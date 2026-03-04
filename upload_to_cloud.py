import json
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time

# --- CONFIGURATION ---
# ⚠️ SECURITY: Ideally, use os.environ.get("PINECONE_API_KEY") 
# But for a student script, you can paste it here temporarily.
PINECONE_API_KEY = "pcsk_7UEHrU_B6tkKMh3SwDUDLgfAtxF68WhmYgL8JaPVbSg53wgctsWdkYMngw3poQF4TewLLF"
INDEX_NAME = "emoji-toxicity"
INPUT_FILE = "data/knowledge_base_enriched.json"

def upload_data():
    print("🚀 Connecting to Pinecone Cloud...")
    if not PINECONE_API_KEY or PINECONE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise RuntimeError("Missing PINECONE_API_KEY. Set env var or update the script.")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists (it should, since you made it in the UI)
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"❌ Error: Index '{INDEX_NAME}' not found. Please create it in the UI first.")
        return

    index = pc.Index(INDEX_NAME)
    
    print("🧠 Loading Embedding Model (all-MiniLM-L6-v2)...")
    # This model creates 384-dimensional vectors
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"📂 Reading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        knowledge_base = json.load(f)

    vectors_to_upload = []
    batch_size = 50 # Upload in chunks to avoid network errors
    
    print(f"⚙️ Processing {len(knowledge_base)} entries...")
    
    for i, entry in enumerate(knowledge_base):
        # Skip safe entries if you want, or include them for context
        if entry.get('risk_category') == 'Safe':
            continue

        # 1. CONSTRUCT THE TEXT TO EMBED
        # This is what the agent "searches" against.
        # We combine the symbol + slang meaning + triggers.
        symbol = entry.get("symbol", "")
        slang_definition = entry.get("slang_definition") or entry.get("slang_meaning", "")
        toxic_signals = entry.get("toxic_signals") or []
        if not symbol:
            continue

        text_to_embed = (
            f"Symbol: {symbol}\n"
            f"Slang Definition: {slang_definition}\n"
            f"Toxic Context Triggers: {', '.join(toxic_signals)}"
        )
        
        # 2. GENERATE VECTOR (The "Embedding")
        vector_values = model.encode(text_to_embed).tolist()
        
        # 3. PREPARE METADATA
        # This is the text payload the Agent gets back.
        metadata = {
            "symbol": symbol,
            "slang_definition": slang_definition,
            "risk_category": entry.get("risk_category", "Unknown"),
            "benign_signals": ", ".join(entry.get("benign_signals") or []),  # Store as string
            "toxic_signals": ", ".join(toxic_signals),  # Store as string
        }
        
        # 4. ADD TO BATCH
        # ID must be a string (e.g., "vec_corn")
        vec_id = "vec_" + "-".join(f"{ord(ch):x}" for ch in symbol)
        vectors_to_upload.append({
            "id": vec_id,
            "values": vector_values, 
            "metadata": metadata
        })
        
        # 5. UPLOAD BATCH
        if len(vectors_to_upload) >= batch_size:
            index.upsert(vectors=vectors_to_upload)
            print(f"   -> Uploaded batch {i}")
            vectors_to_upload = [] # Reset
            time.sleep(0.5) # Be nice to the free tier API

    # Upload leftovers
    if vectors_to_upload:
        index.upsert(vectors=vectors_to_upload)
        print("   -> Uploaded final batch.")

    print("\n✅ Success! Your teammates can now access the database.")

if __name__ == "__main__":
    upload_data()