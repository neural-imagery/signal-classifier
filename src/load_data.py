import numpy as np
import mne
import os
import pandas as pd

def _get_annotations(annot_file):
    """
    Load annotations from a .tsv file.

    Parameters
    ----------
    annot_file : str
        Path to .tsv file.

    Returns
    -------
    annots : mne.Annotations
        Annotations object.
    """

    annots_df = pd.read_csv(annot_file, sep="\t")
    onsets = annots_df.Onset.tolist()
    durations = list(np.array(onsets[1:]) - np.array(onsets[:-1])) + [0]

    annots = mne.Annotations(
        onset=onsets,  # in seconds
        duration=durations,  # in seconds, too
        description=annots_df.trial_type.tolist(),
    )

    return annots

def load_retinotopy_data():
    """
    Load steve's retinotopy data. He looks at left/right checkerboard stimuli.

    Returns
    -------
    data : list of mne.io.Raw
        Intensity data.
    """

    # load all .snirf files in data/Study1/Steve.
    data = []

    folder = "data/Retinotopy/Steve"

    for file in os.listdir(folder):
        if file.endswith(".snirf"):
            snirf_file = os.path.join(folder, file)
            annots_file = snirf_file.replace(".snirf", "_events.tsv")

            nirs = mne.io.read_raw_snirf(snirf_file, preload=True)
            annots = _get_annotations(annots_file)
            nirs.set_annotations(annots)

            data.append(nirs)

    return data
            
            
