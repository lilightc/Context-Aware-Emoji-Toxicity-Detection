"""Orchestrator: merge all data sources into a unified knowledge base with GPT enrichment."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from emoji_toxicity.config import settings, KB_PATH, PROCESSED_DIR, PROJECT_ROOT
from emoji_toxicity.ingestion.cldr import load_cldr_entries
from emoji_toxicity.ingestion.hatemoji import load_hatemoji_entries


SEED_KB_PATH = PROJECT_ROOT / "data" / "knowledge_base_enriched.json"


def _empty_kb_entry(symbol: str, source: str) -> dict:
    """Default KB entry shape, used as a baseline when merging new sources."""
    return {
        "symbol": symbol,
        "literal_meaning": "",
        "slang_meaning": "",
        "slang_definition": "",
        "risk_category": "",
        "toxic_signals": [],
        "benign_signals": [],
        "sources": [source],
    }


def _load_seed_kb() -> dict[str, dict]:
    """Load the existing 509-entry enriched KB as a symbol→entry map."""
    if not SEED_KB_PATH.exists():
        return {}
    with open(SEED_KB_PATH) as f:
        entries = json.load(f)
    return {e["symbol"]: e for e in entries if "symbol" in e}


@lru_cache(maxsize=1)
def _openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _enrich_with_gpt(symbol: str, literal_meaning: str, slang_definition: str = "") -> dict:
    """Use GPT-4o-mini to produce structured enrichment for a single emoji entry."""
    prompt = f"""You are a Content Safety Expert. Analyze this emoji.

Input Emoji: {symbol}
Literal Meaning: {literal_meaning}
Known Slang Definition: {slang_definition or "None"}

Return a JSON object with these exact keys:
{{
    "literal_meaning": "the literal, safe meaning",
    "slang_meaning": "concise explanation of any toxic/slang meaning, or empty string if none",
    "risk_category": "one of: Hate Speech, Sexual, Political, Drug, Safe",
    "toxic_signals": ["5 keywords/phrases suggesting toxic usage"],
    "benign_signals": ["5 keywords/phrases suggesting safe usage"]
}}

Return ONLY valid JSON. No markdown."""

    resp = _openai_client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. No markdown or extra text."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


def build_knowledge_base(
    skip_urban: bool = True,
    skip_silent_signals: bool = True,
    enrich_missing: bool = True,
    max_enrich: int | None = None,
) -> Path:
    """Build the unified knowledge base from all available sources.

    Args:
        skip_urban: Skip Urban Dictionary scraping (slow, rate-limited).
        skip_silent_signals: Skip Silent Signals (may require gated access).
        enrich_missing: Use GPT to enrich entries missing slang data.
        max_enrich: Cap GPT enrichment calls (for budget control). None = no limit.

    Returns:
        Path to the output knowledge_base.jsonl file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Unified map: symbol → merged entry
    kb: dict[str, dict] = {}

    # --- Source 1: Seed KB (existing 509 entries, already enriched) ---
    print("Loading seed knowledge base...")
    seed = _load_seed_kb()
    for sym, entry in seed.items():
        kb[sym] = {
            "symbol": sym,
            "literal_meaning": entry.get("literal_meaning", ""),
            "slang_meaning": entry.get("slang_meaning", ""),
            "slang_definition": entry.get("slang_definition", ""),
            "risk_category": entry.get("risk_category", "Safe"),
            "toxic_signals": entry.get("toxic_signals", []),
            "benign_signals": entry.get("benign_signals", []),
            "sources": ["seed_kb"],
        }
    print(f"  -> {len(kb)} entries from seed KB")

    # --- Source 2: Unicode CLDR ---
    print("Loading CLDR emoji descriptions...")
    cldr_entries = load_cldr_entries()
    for entry in cldr_entries:
        sym = entry["symbol"]
        if sym in kb:
            if not kb[sym]["literal_meaning"]:
                kb[sym]["literal_meaning"] = entry["literal_meaning"]
            if "CLDR" not in kb[sym]["sources"]:
                kb[sym]["sources"].append("CLDR")
        else:
            new_entry = _empty_kb_entry(sym, "CLDR")
            new_entry["literal_meaning"] = entry["literal_meaning"]
            kb[sym] = new_entry
    print(f"  -> {len(cldr_entries)} CLDR entries merged (total: {len(kb)})")

    # --- Source 3: HatemojiBuild ---
    print("Loading HatemojiBuild dataset...")
    try:
        hatemoji_entries = load_hatemoji_entries()
        for entry in hatemoji_entries:
            sym = entry["symbol"]
            if sym in kb:
                kb[sym]["sources"].append("HatemojiBuild")
            else:
                kb[sym] = _empty_kb_entry(sym, "HatemojiBuild")
            kb[sym]["hatemoji_toxic_count"] = entry["toxic_count"]
            kb[sym]["hatemoji_total_count"] = entry["total_count"]
            if entry["examples"]:
                kb[sym]["hatemoji_examples"] = entry["examples"]
        print(f"  -> {len(hatemoji_entries)} HatemojiBuild entries merged (total: {len(kb)})")
    except Exception as e:
        print(f"  [WARN] HatemojiBuild loading failed: {e}")

    # --- Source 4: Silent Signals (optional) ---
    if not skip_silent_signals:
        print("Loading Silent Signals dataset...")
        try:
            from emoji_toxicity.ingestion.silent_signals import load_silent_signals_entries
            ss_entries = load_silent_signals_entries()
            for entry in ss_entries:
                sym = entry["symbol"]
                if sym in kb:
                    if entry.get("slang_meaning"):
                        kb[sym]["slang_meaning"] = (
                            kb[sym]["slang_meaning"] + "; " + entry["slang_meaning"]
                            if kb[sym]["slang_meaning"]
                            else entry["slang_meaning"]
                        )
                    kb[sym]["sources"].append("SilentSignals")
                else:
                    new_entry = _empty_kb_entry(sym, "SilentSignals")
                    new_entry["slang_meaning"] = entry.get("slang_meaning", "")
                    kb[sym] = new_entry
            print(f"  -> {len(ss_entries)} Silent Signals entries merged (total: {len(kb)})")
        except Exception as e:
            print(f"  [WARN] Silent Signals loading failed: {e}")

    # --- Source 5: Urban Dictionary (optional, slow) ---
    if not skip_urban:
        print("Loading Urban Dictionary definitions (this will be slow)...")
        from emoji_toxicity.ingestion.urban_dict import load_urban_dict_entries
        missing = [sym for sym, e in kb.items() if not e.get("slang_definition")]
        ud_entries = load_urban_dict_entries(missing[:100])  # Cap at 100 for sanity
        for entry in ud_entries:
            sym = entry["symbol"]
            if sym in kb:
                kb[sym]["slang_definition"] = entry["slang_definition"]
                kb[sym]["sources"].append("UrbanDictionary")
        print(f"  -> {len(ud_entries)} Urban Dictionary entries merged")

    # --- GPT Enrichment for entries missing structured fields ---
    if enrich_missing:
        needs_enrichment = [
            sym for sym, e in kb.items()
            if not e.get("risk_category") or not e.get("toxic_signals")
        ]
        if max_enrich is not None:
            needs_enrichment = needs_enrichment[:max_enrich]

        if needs_enrichment:
            print(f"Enriching {len(needs_enrichment)} entries with GPT...")
            for sym in tqdm(needs_enrichment, desc="GPT enrichment"):
                entry = kb[sym]
                try:
                    enriched = _enrich_with_gpt(
                        sym,
                        entry.get("literal_meaning", ""),
                        entry.get("slang_definition", ""),
                    )
                    entry["literal_meaning"] = enriched.get("literal_meaning", entry["literal_meaning"])
                    entry["slang_meaning"] = enriched.get("slang_meaning", entry.get("slang_meaning", ""))
                    entry["risk_category"] = enriched.get("risk_category", "Safe")
                    entry["toxic_signals"] = enriched.get("toxic_signals", [])
                    entry["benign_signals"] = enriched.get("benign_signals", [])
                    entry["sources"].append("GPT-enriched")
                except Exception as e:
                    print(f"  [WARN] GPT enrichment failed for {sym}: {e}")

    # --- Write output ---
    print(f"Writing {len(kb)} entries to {KB_PATH}...")
    with open(KB_PATH, "w") as f:
        for entry in kb.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Knowledge base built: {len(kb)} entries -> {KB_PATH}")
    return KB_PATH
