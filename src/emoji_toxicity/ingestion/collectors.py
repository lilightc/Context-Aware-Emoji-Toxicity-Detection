"""Data collectors for dynamic KB updates.

Each collector returns a list of {"text": str, "source": str, "url": str}
dicts suitable for passing to slang_extractor.extract_slang_candidates().

Collectors:
- RedditCollector: scrapes emoji-heavy posts from relevant subreddits
- EmojipediaCollector: monitors Emojipedia for new slang entries
- UserSubmissionCollector: reads user-flagged entries from the Gradio demo
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import requests

from emoji_toxicity.config import settings, DATA_DIR
from emoji_toxicity.utils import extract_emojis

USER_SUBMISSIONS_PATH = DATA_DIR / "user_submissions.jsonl"

# Subreddits known for discussing emoji slang, internet culture, coded language
_SUBREDDITS = [
    "OutOfTheLoop",
    "InternetSlang",
    "GenZ",
    "copypasta",
    "TikTokCringe",
    "youngpeopleyoutube",
]


@dataclass
class CollectionResult:
    """Raw posts collected from a single source."""
    posts: list[dict]
    source: str
    errors: list[str]


def collect_reddit(
    subreddits: list[str] | None = None,
    limit_per_sub: int = 50,
    min_emoji: int = 1,
) -> CollectionResult:
    """Collect emoji-containing posts from Reddit via the public JSON API.

    Uses Reddit's /.json endpoint (no OAuth needed for read-only public data,
    but rate-limited). For production, use PRAW with credentials.

    Args:
        subreddits: List of subreddit names. Defaults to _SUBREDDITS.
        limit_per_sub: Max posts to fetch per subreddit.
        min_emoji: Minimum emoji count to include a post.
    """
    subreddits = subreddits or _SUBREDDITS
    posts = []
    errors = []
    headers = {"User-Agent": "emoji-toxicity-research/1.0"}

    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/hot.json?limit={limit_per_sub}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                errors.append(f"r/{sub}: HTTP {resp.status_code}")
                continue

            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                post_data = child.get("data", {})
                text = post_data.get("title", "") + " " + post_data.get("selftext", "")
                text = text.strip()

                if len(extract_emojis(text)) < min_emoji:
                    continue

                posts.append({
                    "text": text[:500],  # cap length
                    "source": f"reddit/r/{sub}",
                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                })
        except Exception as e:
            errors.append(f"r/{sub}: {e}")

    return CollectionResult(posts=posts, source="reddit", errors=errors)


def collect_emojipedia_recent() -> CollectionResult:
    """Scrape recent emoji slang entries from Emojipedia.

    Targets the Emojipedia slang page for newly documented meanings.
    Falls back gracefully if the page structure changes.
    """
    posts = []
    errors = []

    url = "https://emojipedia.org/emoji-slang"
    headers = {"User-Agent": "emoji-toxicity-research/1.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            errors.append(f"Emojipedia: HTTP {resp.status_code}")
            return CollectionResult(posts=posts, source="emojipedia", errors=errors)

        # Basic regex extraction — Emojipedia pages have emoji + description text
        # This is fragile and should be replaced with proper parsing if used in prod
        emoji_pattern = re.compile(r"([\U00010000-\U0010ffff])\s*[—–-]\s*(.{10,200})")
        for match in emoji_pattern.finditer(resp.text):
            emoji_char, description = match.groups()
            posts.append({
                "text": f"{emoji_char}: {description.strip()}",
                "source": "emojipedia",
                "url": url,
            })
    except Exception as e:
        errors.append(f"Emojipedia: {e}")

    return CollectionResult(posts=posts, source="emojipedia", errors=errors)


def collect_user_submissions() -> CollectionResult:
    """Read user-flagged entries from the Gradio demo.

    Each line in user_submissions.jsonl: {"text": ..., "context": ..., "flag": ...}
    These are messages users flagged as incorrectly classified.
    """
    posts = []
    errors = []

    if not USER_SUBMISSIONS_PATH.exists():
        return CollectionResult(posts=posts, source="user_submissions", errors=[])

    try:
        with open(USER_SUBMISSIONS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                text = entry.get("text", "")
                context = entry.get("context", "")
                if extract_emojis(text):
                    posts.append({
                        "text": f"{text} [context: {context}]" if context else text,
                        "source": "user_submission",
                        "url": "",
                    })
    except Exception as e:
        errors.append(f"user_submissions: {e}")

    return CollectionResult(posts=posts, source="user_submissions", errors=errors)


def collect_all() -> list[dict]:
    """Run all collectors and return merged posts with source metadata."""
    all_posts = []
    all_errors = []

    for name, collector in [
        ("reddit", collect_reddit),
        ("emojipedia", collect_emojipedia_recent),
        ("user_submissions", collect_user_submissions),
    ]:
        print(f"Collecting from {name}...")
        result = collector()
        all_posts.extend(result.posts)
        if result.errors:
            for err in result.errors:
                print(f"  [WARN] {err}")
        print(f"  -> {len(result.posts)} posts")

    print(f"Total collected: {len(all_posts)} posts")
    return all_posts
