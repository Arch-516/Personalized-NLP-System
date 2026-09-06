import json
import os

PROFILE_FILE = "data/user_profile.json"


def create_default_profile():
    return {
        "preferred_style": "simple",
        "preferred_topic": "NLP",
        "preferred_level": "beginner"
    }


def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as file:
            return json.load(file)

    profile = create_default_profile()

    with open(PROFILE_FILE, "w") as file:
        json.dump(profile, file, indent=4)

    return profile


def save_profile(profile):
    with open(PROFILE_FILE, "w") as file:
        json.dump(profile, file, indent=4)


def personalize_response(response, profile):
    if profile["preferred_style"] == "simple":
        return "Simple Explanation:\n" + response
    else:
        return "Detailed Explanation:\n" + response