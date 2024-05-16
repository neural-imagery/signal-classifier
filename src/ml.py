import mne
import numpy as np

def build_train_test(conc, events, event_id):
    epochs = []
    n_epochs = len(conc)
    n_train = round(n_epochs * 0.7)

    for i in range(len(conc)):
        epochs_i = mne.Epochs(
            conc[i],
            events[i],
            event_id=event_id[i],
            tmin=-6,
            tmax=20,
            reject=None,
            # reject_by_annotation=True,
            # proj=True,
            baseline=(None, 0),
            preload=True,
            detrend=None,
            verbose=False,
        )
        epochs.append(epochs_i)

    train_epochs = mne.concatenate_epochs(epochs[:n_train])
    test_epochs = mne.concatenate_epochs(epochs[n_train:])

    return train_epochs, test_epochs

def featurize(epochs, event_id):
    
    # Keep only these channels:
    # ["S2_D3 hbo", "S2_D4 hbo", "S3_D5 hbo", "S3_D6 hbo"]

    new_epochs = epochs.copy()

    new_epochs = new_epochs.pick_channels(["S2_D3 hbo", "S2_D4 hbo", "S3_D5 hbo", "S3_D6 hbo"])

    # Compute the average of the epochs from 0 to 5, 5 to 20, 10 to 15, 15 to 20

    epochs_0_5 = new_epochs.copy().crop(tmin=0, tmax=5).get_data().mean(axis=2)
    epochs_5_20 = new_epochs.copy().crop(tmin=5, tmax=20).get_data().mean(axis=2)
    epochs_10_15 = new_epochs.copy().crop(tmin=10, tmax=15).get_data().mean(axis=2)
    epochs_15_20 = new_epochs.copy().crop(tmin=15, tmax=20).get_data().mean(axis=2)

    # return a matrix for scikit learn
    X = np.concatenate(
        (epochs_0_5, epochs_5_20, epochs_10_15, epochs_15_20), axis=1
    )

    y = epochs.events[:, -1]

    # filter out only the "left" and "right" events
    left_label = event_id["left"]
    right_label = event_id["right"]

    X = X[np.logical_or(y == left_label, y == right_label)]
    y = y[np.logical_or(y == left_label, y == right_label)]

    return X, y

    