import json
import os


# Project root folder
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# User profile file
PROFILE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "user_profile.json"
)


def create_default_profile():
    return {
        "preferred_style": "simple",
        "preferred_topic": "NLP",
        "preferred_level": "beginner"
    }


def load_profile():

    if os.path.exists(PROFILE_FILE):

        with open(
            PROFILE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    profile = create_default_profile()
    save_profile(profile)

    return profile


def save_profile(profile):

    with open(
        PROFILE_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            profile,
            file,
            indent=4
        )


def personalize_response(response, profile):

    style = profile.get(
        "preferred_style",
        "simple"
    )

    if style == "simple":

        return "Simple Explanation:\n" + response

    elif style == "detailed":

        return "Detailed Explanation:\n" + response

    return response