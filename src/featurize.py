import numpy as np
from scipy.stats import linregress

def summary_features(nirs, feature_list, summary_type):
    """
    Perform feature extraction on NIRS data.

    Parameters
    ----------
    nirs : array of shape (n_epochs, n_channels, n_times)
        Processed NIRS data.

    feature_list : list of strings
        List of features to extract. The list can include ``'mean'`` for the
        mean along the summary_type axis, ``'std'`` for standard deviation along the
        time axis and ``'slope'`` for the slope of the linear regression along
        the time axis.

    summary_type : string
        Either "channels" or "time", so that for each epoch, the features are either
        the summary across channels for each time step ("time") or the summary across
        timesteps for each channel ("channels").

    Returns
    -------
    nirs_features : array of shape (n_epochs, (n_channels|n_timesteps)*n_features)
        Features extracted from NIRS data.
    """

    if summary_type == "time":
        axis=1
    elif summary_type == "channels":
        axis=2

    nirs_features = []
    for feature in feature_list:
        if feature == 'mean':
            feature = np.mean(nirs, axis=axis)
        elif feature == 'std':
            feature = np.std(nirs, axis=axis)
        elif feature == 'slope':
            x = range(nirs.shape[2])
            feature = []
            for epoch in nirs:
                ep_slopes = []
                for channel in epoch:
                    ep_slopes.append(linregress(x, channel).slope)
                feature.append(ep_slopes)
        nirs_features.append(feature)

    nirs_features = np.stack(nirs_features, axis=2)
    nirs_features = nirs_features.reshape(len(nirs), -1)  # flatten data

    return nirs_features
