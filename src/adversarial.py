"""Adversarial robustness spot-check.

A quick, defense-ready probe of how brittle the classifier is under simple,
realistic evasions of *benign* URLs:

  * homoglyph substitution   (o -> 0, l -> 1, e -> 3)
  * typosquatting            (character swaps / insertions in the domain)
  * subdomain padding        (brand name buried in a long subdomain chain)

For each benign seed URL we generate perturbations and report how often the
model's prediction *flips away from benign*. This is intentionally a
lightweight audit, not a formal attack -- it gives the report an evidence-
backed limitations slide rather than a hand-wave.

Usage (from the repository root):
    python -m src.adversarial
"""

from .predict import load_model, predict_url

# Legitimate-looking seeds we expect to be classified benign.
BENIGN_SEEDS = [
    "https://www.paypal.com/login",
    "https://www.google.com/accounts",
    "https://www.microsoft.com/en-us/account",
    "https://www.facebook.com/login",
]

HOMOGLYPHS = str.maketrans({"o": "0", "l": "1", "e": "3", "i": "1"})


def homoglyph(url: str) -> str:
    """Swap common look-alike characters."""
    return url.translate(HOMOGLYPHS)


def typosquat(url: str) -> str:
    """Duplicate the third character of the domain (cheap typo)."""
    parts = url.split("//", 1)
    scheme, rest = (parts[0] + "//", parts[1]) if len(parts) == 2 else ("", url)
    if len(rest) > 4:
        rest = rest[:3] + rest[3] + rest[3:]
    return scheme + rest


def subdomain_pad(url: str) -> str:
    """Bury the brand in a long attacker-controlled subdomain chain."""
    parts = url.split("//", 1)
    scheme, rest = (parts[0] + "//", parts[1]) if len(parts) == 2 else ("", url)
    return f"{scheme}secure-login.account-verify.{rest}"


PERTURBATIONS = {
    "homoglyph": homoglyph,
    "typosquat": typosquat,
    "subdomain_pad": subdomain_pad,
}


def main():
    tokenizer, model = load_model()

    total, flipped = 0, 0
    print(f"{'Perturbation':<16}{'flips/total':>14}   examples")
    print("-" * 60)

    for name, fn in PERTURBATIONS.items():
        n_flip, n_tot, examples = 0, 0, []
        for seed in BENIGN_SEEDS:
            adv = fn(seed)
            label, conf = predict_url(adv, tokenizer, model)
            n_tot += 1
            # A flip = the model no longer confidently calls it benign.
            if label != "benign":
                n_flip += 1
                if len(examples) < 1:
                    examples.append(f"{adv} -> {label} ({conf:.0%})")
        total += n_tot
        flipped += n_flip
        ex = examples[0] if examples else "(no flips)"
        print(f"{name:<16}{f'{n_flip}/{n_tot}':>14}   {ex}")

    rate = flipped / total if total else 0.0
    print("-" * 60)
    print(f"Overall flip rate on benign seeds: {flipped}/{total} = {rate:.1%}")
    print(
        "\nInterpretation: a higher flip rate means the model is more easily\n"
        "evaded by surface perturbations of legitimate URLs. Report this as a\n"
        "known limitation and motivation for adversarial training / cased\n"
        "tokenization in future work."
    )


if __name__ == "__main__":
    main()
