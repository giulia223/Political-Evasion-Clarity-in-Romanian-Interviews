"""Label taxonomy (T0.3). Exact strings as used by the platform."""
import numpy as np

EVASION = [
    "Explicit",
    "Implicit",
    "General",
    "Partial/half-answer",
    "Dodging",
    "Deflection",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]
CLARITY = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

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

EV2I = {l: i for i, l in enumerate(EVASION)}
CL2I = {l: i for i, l in enumerate(CLARITY)}
# CLARITY_IDX_OF[e] = clarity index of evasion index e
CLARITY_IDX_OF = np.array([CL2I[CLARITY_OF[e]] for e in EVASION])
# GROUP[c, e] = 1 if evasion e belongs to clarity c
GROUP = np.zeros((len(CLARITY), len(EVASION)))
GROUP[CLARITY_IDX_OF, np.arange(len(EVASION))] = 1.0

EV_COLS = [f"p_ev_{l}" for l in EVASION]
CL_COLS = [f"p_cl_{l}" for l in CLARITY]


def clarity_of(evasion_labels):
    """Map evasion label strings to clarity label strings."""
    return [CLARITY_OF[e] for e in evasion_labels]
