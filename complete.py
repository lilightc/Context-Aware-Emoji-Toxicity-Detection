import json
import os

try:
    from openai import OpenAI
except Exception: 
    OpenAI = None

# 1. Load your raw scraped data
with open("data/slang_dictionary_complete.json", "r") as f:
    raw_data = json.load(f)

structured_knowledge_base = []

# 2. Iterate through raw entries and "Enrich" them
def call_llm(prompt: str) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run `pip install openai`.")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. No markdown or extra text.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

for i, entry in enumerate(raw_data):
    symbol = entry['symbol']
    raw_def = entry.get('slang_definition', '')

    # --- THE UPDATED PROMPT ---
    # Explicitly asking for "symbol" in the output JSON
    prompt = f"""
    You are a Content Safety Expert. Analyze this emoji context.
    
    Input Emoji: {symbol}
    Input Definition: "{raw_def}"
    
    Task:
    1. "literal_meaning": The literal, safe meaning (e.g. fruit, animal).
    2. "slang_meaning": A concise explanation of the toxic meaning.
    3. "toxic_signals": A list of 5 keywords/phrases that suggest the TOXIC meaning (e.g. "link in bio").
    4. "benign_signals": A list of 5 keywords/phrases that suggest the SAFE meaning.
    5. "risk_category": One of [Hate Speech, Sexual, Political, Drug, Safe].

    Return ONLY a JSON object with these exact keys:
    {{
        "symbol": "{symbol}", 
        "literal_meaning": "...",
        "slang_meaning": "...",
        "risk_category": "...",
        "toxic_signals": ["..."],
        "benign_signals": ["..."]
    }}
    """
    
    # ... (Call LLM API here) ...
    response = call_llm(prompt)
    structured_entry = json.loads(response)
    
    structured_knowledge_base.append(structured_entry)

# 3. Save the "Golden" Knowledge Base
with open("data/knowledge_base_enriched.json", "w") as f:
    json.dump(structured_knowledge_base, f, indent=2)