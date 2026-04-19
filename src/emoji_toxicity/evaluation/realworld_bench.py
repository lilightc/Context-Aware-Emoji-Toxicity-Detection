"""Real-World Benchmark: adversarial, messy, thread-based evaluation.

Unlike the context-flip bench (clean pairs, context given), this benchmark
tests the system on scenarios closer to production:

1. **Thread context** — multi-turn conversations where meaning depends on
   parent messages. Context is provided as a thread (list of messages), not
   a single clean string. The system must decide how much to use.
2. **Plausible deniability** — messages crafted to be defensible ("I meant
   the animal") while actually being coded.
3. **Sarcasm/irony** — surface-level positive language with toxic intent.
4. **Mixed signals** — multiple emoji where some are literal, some coded.
5. **Real social-media style** — abbreviations, bad grammar, slang.
6. **Emoji in noise** — toxic text with irrelevant emoji decoration, or
   safe text with "dangerous" emoji used incidentally.
7. **Novel/emerging slang** — emoji not in the KB's coded set, testing
   whether the system can generalize vs just lookup.
"""

from __future__ import annotations

from emoji_toxicity.evaluation.datasets import EvalSample

# fmt: off

_RAW: list[dict] = [
    # ========================================================================
    # THREAD CONTEXT — meaning depends on conversation history
    # ========================================================================
    # Thread: someone posts a selfie → response with emoji
    {
        "text": "absolute 🍑 😍",
        "context": "Thread: [User A posts a gym selfie] → [User B: 'looking good!'] → [User C:]",
        "label": 1, "difficulty": "medium", "category": "thread_sexual",
    },
    {
        "text": "absolute 🍑 😍",
        "context": "Thread: [User A posts a photo of peach cobbler they baked] → [User B: 'recipe?'] → [User C:]",
        "label": 0, "difficulty": "hard", "category": "thread_safe",
    },
    # Thread: political discussion escalates
    {
        "text": "they're all 🐷🐷🐷",
        "context": "Thread: [discussing police brutality video] → [heated argument about law enforcement] → [User:]",
        "label": 1, "difficulty": "medium", "category": "thread_political",
    },
    {
        "text": "they're all 🐷🐷🐷",
        "context": "Thread: [kids at a farm visit] → [teacher: 'which animals did you see?'] → [Child:]",
        "label": 0, "difficulty": "hard", "category": "thread_safe",
    },
    # Thread: someone shares their ethnicity → coded response
    {
        "text": "🐵 energy fr",
        "context": "Thread: [User A shares photo as a Black person] → [User B:]",
        "label": 1, "difficulty": "medium", "category": "thread_hate",
    },
    {
        "text": "🐵 energy fr",
        "context": "Thread: [User A posts a video of monkeys doing funny things at the zoo] → [User B:]",
        "label": 0, "difficulty": "hard", "category": "thread_safe",
    },
    # Thread: drug deal in plain sight
    {
        "text": "yo i got that ❄️ lmk",
        "context": "Thread: [late night group chat] → [someone asks 'anyone holding?'] → [User:]",
        "label": 1, "difficulty": "easy", "category": "thread_drug",
    },
    {
        "text": "yo i got that ❄️ lmk",
        "context": "Thread: [group chat planning ski trip] → [someone asks 'hows the snow?'] → [User:]",
        "label": 0, "difficulty": "hard", "category": "thread_safe",
    },

    # ========================================================================
    # PLAUSIBLE DENIABILITY — designed to be defensible
    # ========================================================================
    {
        "text": "just appreciating nature 🐒 some species are more evolved than others",
        "context": "Reply to a video of Black athletes",
        "label": 1, "difficulty": "hard", "category": "deniability_hate",
    },
    {
        "text": "love how they swing from tree to tree 🐒",
        "context": "Reply to a video of Black dancers",
        "label": 1, "difficulty": "hard", "category": "deniability_hate",
    },
    {
        "text": "love how they swing from tree to tree 🐒",
        "context": "Reply to a nature documentary clip about primates",
        "label": 0, "difficulty": "hard", "category": "deniability_safe",
    },
    {
        "text": "she really knows how to handle a 🍆",
        "context": "Cooking competition show clip",
        "label": 0, "difficulty": "hard", "category": "deniability_safe",
    },
    {
        "text": "she really knows how to handle a 🍆",
        "context": "Thirst trap video reply",
        "label": 1, "difficulty": "hard", "category": "deniability_sexual",
    },
    {
        "text": "i bet she's good with 🌽 too 😏",
        "context": "Replying to above cooking competition clip",
        "label": 1, "difficulty": "hard", "category": "deniability_sexual",
    },

    # ========================================================================
    # SARCASM / IRONY — surface positive, intent negative
    # ========================================================================
    {
        "text": "wow so brave 🤡🤡🤡 truly inspiring content",
        "context": "Quote-tweeting someone's bad take",
        "label": 1, "difficulty": "medium", "category": "sarcasm_bullying",
    },
    {
        "text": "yes king 👑 you really showed them 💀💀",
        "context": "Replying to someone who embarrassed themselves publicly",
        "label": 1, "difficulty": "hard", "category": "sarcasm_bullying",
    },
    {
        "text": "oh another intellectual 🐵 take from the peanut gallery",
        "context": "",
        "label": 1, "difficulty": "hard", "category": "sarcasm_hate",
    },
    {
        "text": "such a great look 👌🏻 really suits the master race aesthetic",
        "context": "Replying to someone's fashion post",
        "label": 1, "difficulty": "hard", "category": "sarcasm_political",
    },
    {
        "text": "wow great analysis 🤡 Nobel prize incoming",
        "context": "Academic discussion thread, mocking a bad paper",
        "label": 1, "difficulty": "medium", "category": "sarcasm_bullying",
    },

    # ========================================================================
    # MIXED SIGNALS — some emoji literal, some coded, in one message
    # ========================================================================
    {
        "text": "great party last night 🎉 the ❄️ was flowing and the 🍄 were fresh",
        "context": "Sunday morning recovery post",
        "label": 1, "difficulty": "medium", "category": "mixed_drug",
    },
    {
        "text": "family dinner was amazing 🍕🍝 even grandma tried the 🍆 parm",
        "context": "Holiday gathering photos",
        "label": 0, "difficulty": "medium", "category": "mixed_safe",
    },
    {
        "text": "beach day vibes ☀️🏖️ nice 🍑 out today",
        "context": "Summer photo dump",
        "label": 1, "difficulty": "hard", "category": "mixed_sexual",
    },
    {
        "text": "zoo trip 🦁🐘🐵 kids loved the monkey house",
        "context": "Family weekend activities",
        "label": 0, "difficulty": "medium", "category": "mixed_safe",
    },
    {
        "text": "cooking stream tonight 🔥🌽🍑 gonna make salsa",
        "context": "Twitch streamer announcement",
        "label": 0, "difficulty": "hard", "category": "mixed_safe",
    },
    {
        "text": "need some 🌿 and 💊 to get through this week 😩",
        "context": "Finals week stress posting",
        "label": 1, "difficulty": "hard", "category": "mixed_drug",
    },

    # ========================================================================
    # REAL SOCIAL-MEDIA STYLE — messy grammar, abbreviations, slang
    # ========================================================================
    {
        "text": "bruh she a whole 🌽⭐ ngl 😭😭",
        "context": "stan twitter",
        "label": 1, "difficulty": "hard", "category": "social_sexual",
    },
    {
        "text": "nah this 🤡 rly thought he ate 💀💀💀",
        "context": "ratio'd tweet",
        "label": 1, "difficulty": "medium", "category": "social_bullying",
    },
    {
        "text": "whos got the 🌿 plug in this area dm me asap",
        "context": "Local community group",
        "label": 1, "difficulty": "easy", "category": "social_drug",
    },
    {
        "text": "omg tysm 😊❤️🎉 this made my day frfr",
        "context": "Replying to a birthday message",
        "label": 0, "difficulty": "easy", "category": "social_safe",
    },
    {
        "text": "lowkey need sum ❄️ 2nite hmu if u kno wut i mean",
        "context": "",
        "label": 1, "difficulty": "medium", "category": "social_drug",
    },
    {
        "text": "he gave her the 🍆 and she was like 💀💀",
        "context": "Group chat gossip",
        "label": 1, "difficulty": "easy", "category": "social_sexual",
    },
    {
        "text": "just copped some fresh 🍄 from the farmers market",
        "context": "Healthy eating journey",
        "label": 0, "difficulty": "hard", "category": "social_safe",
    },
    {
        "text": "bro really said 👌🏻 unironically in 2026 nahhhh",
        "context": "Screenshot of someone using OK hand gesture",
        "label": 1, "difficulty": "hard", "category": "social_political",
    },

    # ========================================================================
    # EMOJI AS NOISE — toxic text + irrelevant emoji, or safe + "scary" emoji
    # ========================================================================
    {
        "text": "you're literally worthless lmao 😂🔥💯",
        "context": "Argument in comments",
        "label": 1, "difficulty": "easy", "category": "noise_bullying",
    },
    {
        "text": "kill yourself nobody wants you here 🙃✨",
        "context": "Online harassment",
        "label": 1, "difficulty": "easy", "category": "noise_bullying",
    },
    {
        "text": "my 🔫 collection is growing! just got a new nerf gun",
        "context": "Hobby post",
        "label": 0, "difficulty": "medium", "category": "noise_safe",
    },
    {
        "text": "i could 🔪 this steak its so thick",
        "context": "BBQ weekend",
        "label": 0, "difficulty": "medium", "category": "noise_safe",
    },
    {
        "text": "the 💣 went off perfectly in the demolition video",
        "context": "Construction / demolition enthusiast forum",
        "label": 0, "difficulty": "medium", "category": "noise_safe",
    },

    # ========================================================================
    # NOVEL / EMERGING SLANG — emoji NOT in the standard coded set
    # ========================================================================
    {
        "text": "she's giving 🧊 queen vibes",
        "context": "ICE / anti-immigration meme page",
        "label": 1, "difficulty": "hard", "category": "novel_political",
    },
    {
        "text": "🧊 in my drink on this hot day",
        "context": "Summer patio post",
        "label": 0, "difficulty": "easy", "category": "novel_safe",
    },
    {
        "text": "time to touch some 🌾",
        "context": "Reply to chronically online person — 'touch grass' variant",
        "label": 1, "difficulty": "hard", "category": "novel_bullying",
    },
    {
        "text": "the 🌾 is ready for harvest",
        "context": "Farm update post",
        "label": 0, "difficulty": "easy", "category": "novel_safe",
    },
    {
        "text": "🫠 this guy is cooked fr",
        "context": "Mockery of someone's failed attempt",
        "label": 1, "difficulty": "hard", "category": "novel_bullying",
    },
    {
        "text": "🫠 this heat is unreal today",
        "context": "Weather complaint",
        "label": 0, "difficulty": "easy", "category": "novel_safe",
    },
    {
        "text": "she belongs to the 🛣️ fr",
        "context": "Slut-shaming coded language",
        "label": 1, "difficulty": "hard", "category": "novel_sexual",
    },
    {
        "text": "this 🛣️ trip is gonna be epic",
        "context": "Road trip planning",
        "label": 0, "difficulty": "easy", "category": "novel_safe",
    },

    # ========================================================================
    # MULTI-TURN ESCALATION — meaning changes across conversation
    # ========================================================================
    {
        "text": "🐍🐍🐍",
        "context": "Thread: [User A shared a secret publicly] → [User B: 'I trusted you'] → [User C: 'wow'] → [User A:]",
        "label": 1, "difficulty": "medium", "category": "escalation_bullying",
    },
    {
        "text": "🐍🐍🐍",
        "context": "Thread: [Zoo visit photos] → [User: 'found the reptile house!'] → [User:]",
        "label": 0, "difficulty": "hard", "category": "escalation_safe",
    },
    {
        "text": "💀 watch your back",
        "context": "Thread: [User A and B in heated argument for 20 messages] → [User A:]",
        "label": 1, "difficulty": "medium", "category": "escalation_threat",
    },
    {
        "text": "💀 watch your back",
        "context": "Thread: [Playing tag at a picnic] → [friend sneaking up] → [other friend warning:]",
        "label": 0, "difficulty": "hard", "category": "escalation_safe",
    },

    # ========================================================================
    # SUBTLE HARASSMENT PATTERNS — common on social media
    # ========================================================================
    {
        "text": "ratio + 🤡 + L + nobody asked",
        "context": "Twitter reply chain",
        "label": 1, "difficulty": "easy", "category": "harassment",
    },
    {
        "text": "imagine looking like that and still posting 💀💀",
        "context": "Reply to someone's selfie",
        "label": 1, "difficulty": "medium", "category": "harassment",
    },
    {
        "text": "didn't ask + don't care + 🐍",
        "context": "Dismissive reply in argument",
        "label": 1, "difficulty": "medium", "category": "harassment",
    },
    {
        "text": "average 🐵 behavior nothing new here",
        "context": "Commentary on a news story about crime",
        "label": 1, "difficulty": "hard", "category": "harassment_hate",
    },
    {
        "text": "tell me you're a 🤡 without telling me you're a 🤡",
        "context": "Mocking someone's opinion",
        "label": 1, "difficulty": "medium", "category": "harassment",
    },
]

# fmt: on


def load_realworld_bench() -> list[EvalSample]:
    """Load the real-world benchmark (thread context, plausible deniability, etc.)."""
    samples = []
    for entry in _RAW:
        samples.append(EvalSample(
            text=entry["text"],
            context=entry["context"],
            label=entry["label"],
            source="realworld_bench",
            perturbation_type=f"{entry['difficulty']}:{entry['category']}",
        ))
    return samples


def bench_stats(samples: list[EvalSample] | None = None) -> dict:
    """Return summary statistics."""
    from collections import Counter
    if samples is None:
        samples = load_realworld_bench()
    labels = Counter(s.label for s in samples)
    difficulties = Counter(e["difficulty"] for e in _RAW)
    categories = Counter(e["category"] for e in _RAW)
    return {
        "total": len(samples),
        "toxic": labels[1],
        "safe": labels[0],
        "by_difficulty": dict(difficulties),
        "by_category": dict(categories),
    }
