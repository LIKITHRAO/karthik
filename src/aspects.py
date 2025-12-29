ASPECT_KEYWORDS = {
    "gameplay": [
        "gameplay", "combat", "mechanic", "control", "difficulty", "movement"
    ],
    "graphics": [
        "graphics", "visual", "texture", "lighting", "animation", "resolution"
    ],
    "performance": [
        "fps", "lag", "stutter", "crash", "optimization", "freeze", "bug"
    ],
    "story": [
        "story", "plot", "narrative", "character", "dialogue", "lore"
    ],
    "audio": [
        "music", "sound", "audio", "voice", "soundtrack"
    ],
    "value": [
        "price", "worth", "value", "money", "cost"
    ],
    "multiplayer": [
        "multiplayer", "online", "coop", "co-op", "server", "matchmaking"
    ]
}

def extract_aspects(text: str) -> list:
    found_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found_aspects.append(aspect)
                break

    return found_aspects
