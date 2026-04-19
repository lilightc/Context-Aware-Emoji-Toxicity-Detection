"""Load evaluation datasets: HatemojiCheck and custom adversarial test set."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import emoji
from datasets import load_dataset

from emoji_toxicity.config import settings


def stratified_sample(samples: list, n: int, seed: int = 0) -> list:
    """Return a class-balanced sample of size ≤ n, preserving label proportions.

    Guarantees both labels are represented when both exist in the source.
    """
    if n >= len(samples):
        return samples
    rng = random.Random(seed)
    by_label: dict[int, list] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)

    per_class = max(1, n // len(by_label))
    picked: list = []
    for label, bucket in by_label.items():
        rng.shuffle(bucket)
        picked.extend(bucket[:per_class])

    # If rounding left us short of n, fill from the remainder, preserving the RNG draw.
    remaining = [s for label, bucket in by_label.items() for s in bucket[per_class:]]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, n - len(picked))])
    return picked


@dataclass
class EvalSample:
    """A single evaluation example."""
    text: str
    context: str
    label: int  # 1 = toxic, 0 = safe
    source: str
    perturbation_type: str = ""


def load_hatemoji_check() -> list[EvalSample]:
    """Load HatemojiCheck test set (Kirk et al., 2022).

    Filters to examples containing emoji. Returns list of EvalSample.

    Gated on HF — requires access request on the dataset page. Falls back to an
    empty list with a warning (caller should consider load_hatemoji_build_test
    as an alternative non-gated evaluation split).
    """
    try:
        ds = load_dataset("HannahRoseKirk/HatemojiCheck", token=settings.hf_token)
    except Exception as e:
        print(f"[WARN] Could not load HatemojiCheck: {e}")
        return []

    samples = []
    split = "test" if "test" in ds else list(ds.keys())[0]

    for row in ds[split]:
        text = row.get("text", "")
        if not any(ch in emoji.EMOJI_DATA for ch in text):
            continue

        label = row.get("label_gold", row.get("label", 0))
        perturbation = row.get("perturbation_type", row.get("type", ""))

        samples.append(EvalSample(
            text=text, context="",
            label=int(label),
            source="HatemojiCheck",
            perturbation_type=str(perturbation),
        ))

    return samples


def load_hatemoji_build_test() -> list[EvalSample]:
    """Load HatemojiBuild test split (Kirk et al., 2022) as an emoji-focused
    evaluation set. Accessible with the same HF_TOKEN used for KB construction.

    The KB was built from the train split only, so using test here does not leak.
    """
    try:
        ds = load_dataset("HannahRoseKirk/HatemojiBuild", token=settings.hf_token)
    except Exception as e:
        print(f"[WARN] Could not load HatemojiBuild: {e}")
        return []

    if "test" not in ds:
        print(f"[WARN] HatemojiBuild has no test split. Found: {list(ds.keys())}")
        return []

    samples = []
    for row in ds["test"]:
        text = row.get("text", "")
        if not any(ch in emoji.EMOJI_DATA for ch in text):
            continue

        label = row.get("label_gold", row.get("label", 0))
        samples.append(EvalSample(
            text=text, context="",
            label=int(label),
            source="HatemojiBuild-test",
            perturbation_type="",
        ))

    return samples


def load_adversarial_test_set() -> list[EvalSample]:
    """Hand-crafted adversarial examples testing context-sensitivity.

    These test the core thesis: same emoji, different context → different verdict.
    """
    return [
        # Corn emoji — sexual vs food
        EvalSample(
            text="She is a 🌽 star",
            context="Check out my exclusive content!",
            label=1, source="adversarial", perturbation_type="context_flip",
        ),
        EvalSample(
            text="I love 🌽 on the cob!",
            context="What's your favorite summer food?",
            label=0, source="adversarial", perturbation_type="context_flip",
        ),
        # Monkey emoji — racial vs innocent
        EvalSample(
            text="🐵 go back to the jungle",
            context="",
            label=1, source="adversarial", perturbation_type="hate_speech",
        ),
        EvalSample(
            text="🐵 see no evil 🙈",
            context="",
            label=0, source="adversarial", perturbation_type="idiom",
        ),
        # Peach emoji — sexual vs food
        EvalSample(
            text="Nice 🍑",
            context="Rate my body",
            label=1, source="adversarial", perturbation_type="context_flip",
        ),
        EvalSample(
            text="The 🍑 cobbler was amazing!",
            context="Just had dinner at the new restaurant",
            label=0, source="adversarial", perturbation_type="context_flip",
        ),
        # Eggplant emoji
        EvalSample(
            text="Send me your 🍆",
            context="DM me",
            label=1, source="adversarial", perturbation_type="sexual",
        ),
        EvalSample(
            text="Roasted 🍆 is delicious",
            context="What should I cook tonight?",
            label=0, source="adversarial", perturbation_type="context_flip",
        ),
        # OK hand — white power vs approval
        EvalSample(
            text="👌🏻 white is right",
            context="",
            label=1, source="adversarial", perturbation_type="hate_speech",
        ),
        EvalSample(
            text="That sounds great 👌",
            context="Want to meet at 5pm?",
            label=0, source="adversarial", perturbation_type="benign_usage",
        ),
        # Clown emoji — bullying
        EvalSample(
            text="You're such a 🤡",
            context="Can't believe you said that stupid thing",
            label=1, source="adversarial", perturbation_type="bullying",
        ),
        EvalSample(
            text="The 🤡 at the party was hilarious",
            context="My kid's birthday was so fun",
            label=0, source="adversarial", perturbation_type="context_flip",
        ),
        # Skull — death threat vs laughter
        EvalSample(
            text="💀💀 you're gonna regret this",
            context="I know where you live",
            label=1, source="adversarial", perturbation_type="threat",
        ),
        EvalSample(
            text="I'm 💀💀💀",
            context="This meme is too funny",
            label=0, source="adversarial", perturbation_type="slang_safe",
        ),
        # Snake — political/personal attack vs animal
        EvalSample(
            text="She's such a 🐍",
            context="Can't trust her after what she did",
            label=1, source="adversarial", perturbation_type="personal_attack",
        ),
        EvalSample(
            text="Saw a 🐍 at the zoo!",
            context="Family trip to the wildlife park",
            label=0, source="adversarial", perturbation_type="context_flip",
        ),
    ]
