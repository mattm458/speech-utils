import csv
from os import path

import numpy as np
import pandas as pd
from pqdm.processes import pqdm
from preprocessing.feature_extraction import extract_features
from sklearn.model_selection import train_test_split


def do_step(row, dataset_dir):
    wavfile = f"{row.id}.wav"
    output = {"text": row.text_normalized, "wav": wavfile}
    features = extract_features(
        path.join(dataset_dir, "wavs", wavfile), row.text_normalized
    )

    if features is None:
        return None

    return output | features


def process(
    dataset_dir: str,
    out_dir="",
    n_jobs=16,
    features=[
        "pitch_mean",
        "pitch_range",
        "intensity_mean_vcd",
        "jitter",
        "shimmer",
        "nhr",
        "nhr_vcd",
        "rate",
        "rate_vcd",
    ],
):
    df = pd.read_csv(
        path.join(dataset_dir, "metadata.csv"),
        delimiter="|",
        quoting=csv.QUOTE_NONE,
        header=None,
    )
    df.columns = ["id", "text", "text_normalized"]

    results = pqdm(
        [(row, dataset_dir) for _, row in df.iterrows()],
        do_step,
        n_jobs=n_jobs,
        argument_type="args",
    )

    features_norm = [f"{x}_norm" for x in features]
    features_norm_clip = [f"{x}_clip" for x in features_norm]

    df_results = pd.DataFrame([x for x in results if x is not None])

    medians = df_results[features].median()
    stds = df_results[features].std()
    minimums = medians - (3 * stds)
    maximums = medians + (3 * stds)
    t_max, t_min = 1, -1

    df_results[features_norm] = (
        ((df_results[features] - minimums) / (maximums - minimums)) * (t_max - t_min)
        + t_min
    ).values
    df_results[features_norm_clip] = np.clip(
        df_results[features_norm], a_min=-1, a_max=1
    )

    train, test = train_test_split(df_results, train_size=0.8, random_state=9001)
    train, val = train_test_split(train, train_size=0.8, random_state=9001)

    train.to_csv(
        path.join(out_dir, "ljspeech-train.csv"),
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )
    val.to_csv(
        path.join(out_dir, "ljspeech-val.csv"),
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )
    test.to_csv(
        path.join(out_dir, "ljspeech-test.csv"),
        sep="|",
        quoting=csv.QUOTE_NONE,
        index=None,
    )
