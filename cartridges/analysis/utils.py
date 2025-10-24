

COLORS = list(map(lambda x: f"#{x}", [
    "3F82E0",
    "FE761E",
    "EB2432",
    "357389",
    "ff758f",
    "57cc99",
    "FEC426",
    "B147DA",
    "eae2b7"
]))






HUE_TO_NAME = {
    "cartridge": "Cartridges",
    "icl": "ICL",
    "first_k_tokens": "First K Tokens (Prompt Compression)",
    "duo": "Duo Attention (KV-Cache Compression)",
    "summary": "Summary (Prompt Compression)",
    "ntp": "Next-Token Prediction",
}


HUE_TO_COLOR = {
    "cartridge": COLORS[0],
    "cartridge_composition": COLORS[0],
    "icl": COLORS[1],
    "first_k_tokens": COLORS[2],
    "duo": COLORS[3],
    "summary": COLORS[4],
    "ntp": COLORS[5],
    "lora_rank": "#57cc99",
}

HUE_TO_GRADIENTS = {
    "cartridge": ("#3F82E0", "#E1F5FF"),
    "icl": ("#FE761E", "#FE761E"),
    "lora_rank": ("#57cc99", "#a4f5d2")
}

GRADIENT_COLORS = [
    COLORS[0], 
    "#5A97E6",
    "#75ACEC",
    "#90C1F2",
    "#ABD6F8",
    "#C6EBFE",
    "#E1F5FF"
]


BACKUP_COLORS = COLORS[len(HUE_TO_COLOR):]


HUE_TO_ZORDER = {
    "cartridge": 100,
    "icl": 90,
    "first_k_tokens": 80,
    "duo": 70,
    "summary": 60,
    "lora_rank": 60,
}

