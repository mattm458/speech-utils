from typing import Dict, Optional

import hyphenate
import parselmouth
from nltk.corpus import cmudict

MIN_DUR = 6.4 / 75
d = cmudict.dict()

with open("preprocessing/processing.praat") as infile:
    script = infile.read()


def extract_features(
    wavfile: str,
    transcript: str,
    start: Optional[float] = None,
    end: Optional[float] = None,
    channel: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    sound = parselmouth.Sound(wavfile)

    if channel is not None:
        sound = sound.extract_channel(channel=channel)

    if start is not None or end is not None:
        sound = sound.extract_part(from_time=start, to_time=end)

    features = get_features(sound=sound)

    if features is None:
        return None

    num_syllables = sum([get_syllables(word) for word in transcript.split(" ")])

    features['rate'] = features['duration'] / num_syllables
    features['rate_vcd'] = features['duration_vcd'] / num_syllables

    return features


def get_features(sound: parselmouth.Sound) -> Optional[Dict]:
    try:
        _, output = parselmouth.praat.run(sound, script, capture_output=True)

        features_in = [tuple(x.split(",")) for x in output.split("\n") if x]
        features_in = dict([(k, float(v)) for k, v in features_in])
        return features_in
    except:
        return None


def get_syllables(word: str) -> int:
    # Remove trailing whitespace and convert to lowercase for dictionary lookup.
    word = word.strip().lower()

    # Special case: Empty string.
    if len(word) == 0:
        return 0

    if word == "((" or word == "))":
        return 0

    if word == "[noise]" or word == "[laughter]":
        return 0

    # Special case: Words consisting only of "?"
    # This situation occurs when the utterance is unclear, and annotators leave
    # one question mark per syllable.
    num_qs = sum([1 if x == "?" else 0 for x in word])
    if num_qs == len(word):
        return num_qs

    # Special case: If the word ends in "-", remove the dash and try to determine
    # syllables as best we can. This occurs if the speaker does not complete the word.
    if word[-1] == "-":
        word = word[:-1]

    # Special case: If there is an apostrophe in the word, then it may not be
    # in the dictionary.
    if "'" in word:
        # A common situation is "'s", where the dictionary does not contain the possessive
        # form of all words. If that applies here, remove the "'s" and look up the
        # singular form of the word.
        if word not in d and word[-2:] == "'s":
            word = word[:-2]

    # Special case: Neither the dictionary or automatic hyphenation know what an
    # M&M is. If possessive ("m&m's"), it was made singular above.
    if word == "m&m" or word == "m&ms":
        return 3

    # Main syllable lookup functionality.
    if word in d:
        # If the word is in the dictionary, extract the syllable count.
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word]][0]
    else:
        # Otherwise, fall back to the hyphenate library for a best (but
        # sometimes inaccurate) guess.
        return len(hyphenate.hyphenate_word(word))
