"""Label taxonomy (T0.3): 9 evasion labels, 3 clarity labels, and the fixed upward map."""
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

EV2ID = {l: i for i, l in enumerate(EVASION)}
CL2ID = {l: i for i, l in enumerate(CLARITY)}
# clarity index of each evasion index
EV_TO_CL_IDX = [CL2ID[CLARITY_OF[l]] for l in EVASION]

EV_COLS = [f"p_ev_{l}" for l in EVASION]
CL_COLS = [f"p_cl_{l}" for l in CLARITY]
