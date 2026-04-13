"""Load evaluation datasets: HatemojiCheck and custom adversarial test set."""

from __future__ import annotations

from dataclasses import dataclass

import emoji
from datasets import load_dataset

from emoji_toxicity.config import settings


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
        # Only include examples that contain emoji
        has_emoji = any(ch in emoji.EMOJI_DATA for ch in text)
        if not has_emoji:
            continue

        label = row.get("label_gold", row.get("label", 0))
        perturbation = row.get("perturbation_type", row.get("type", ""))

        samples.append(EvalSample(
            text=text,
            context="",
            label=int(label),
            source="HatemojiCheck",
            perturbation_type=str(perturbation),
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
