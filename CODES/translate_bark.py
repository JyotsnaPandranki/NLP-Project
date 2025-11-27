import random

label_to_phrases = {
    "CH-N": [
        "I feel uncomfortable...",
        "Something isn’t right.",
        "Please stop that."
    ],
    "CH-P": [
        "Yay this is fun!",
        "I'm happy!",
        "Let’s keep going!"
    ],
    "GR-N": [
        "Back off!",
        "Stay away from me!",
        "Don’t come closer!"
    ],
    "GR-P": [
        "That’s mine!",
        "Play with me!",
        "I'm excited!"
    ],
    "L-A": [
        "Someone is here!",
        "Alert alert!",
        "What’s that sound?!"
    ],
    "L-D": [
        "I’m not liking this!",
        "Stop that!",
        "I’m upset!"
    ],
    "L-H": [
        "Welcome home!!!",
        "I missed you!",
        "Yay you’re here!"
    ],
    "L-O": [
        "Huh? What’s that?",
        "So curious...",
        "Let me check that out."
    ],
    "L-P": [
        "Let’s play! Now!",
        "Throw the ball!",
        "I’m excited!"
    ],
    "L-PA": [
        "Outside pls! Walk time!",
        "Let’s go out!",
        "I want fresh air!"
    ],
    "L-S": [
        "Look there!",
        "Hey hey hey!",
        "Something’s happening!"
    ],
    "L-S1": [
        "Who’s that?",
        "Someone outside!",
        "What’s going on?!"
    ],
    "L-S2": [
        "Stay away from here!",
        "This is my place!",
        "Don’t try anything!"
    ],
    "L-S3": [
        "I’m scared...",
        "I don’t like that...",
        "Please leave me alone..."
    ],
    "L-TA": [
        "Don’t leave me 😢",
        "Where are you going?",
        "Please come back!"
    ],
    "L-W": [
        "Warning!!",
        "Danger!",
        "Back off right now!"
    ],
    "S": [
        "(just breathing)",
        "Nothing special right now",
        "Quiet time..."
    ],
}

def translate_label(label, confidence=1.0):
    phrases = label_to_phrases.get(label, ["I’m confused..."])
    phrase = random.choice(phrases)

    if confidence > 0.85 and not phrase.endswith("!"):
        phrase += "!"
    elif confidence < 0.45 and not phrase.endswith("..."):
        phrase += "..."

    return phrase
