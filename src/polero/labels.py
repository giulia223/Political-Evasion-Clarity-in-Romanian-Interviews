"""Label taxonomy: 9 evasion labels, 3 clarity labels and the fixed upward mapping."""

CLARITY = ("Clear Reply", "Ambivalent", "Clear Non-Reply")

CLARITY_OF = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}

EVASION = tuple(CLARITY_OF)

EVASION_TO_ID = {label: i for i, label in enumerate(EVASION)}
CLARITY_TO_ID = {label: i for i, label in enumerate(CLARITY)}

# EVASION_GROUP[i] is the clarity index of evasion index i (used for summing probabilities per group).
EVASION_GROUP = tuple(CLARITY_TO_ID[CLARITY_OF[e]] for e in EVASION)


def to_clarity(evasion_labels):
    """Map evasion label strings to clarity label strings."""
    return [CLARITY_OF[e] for e in evasion_labels]
