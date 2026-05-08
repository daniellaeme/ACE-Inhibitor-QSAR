import numpy as np


def check_domain(fingerprint, pca_model, centroid, threshold):
    """
    Projects a new molecule's fingerprint into the trained PCA space.
    checks if it falls within the Applicability Domain.

    Args:
        fingerprint (np.array): The Morgan Fingerprint.
        pca_model (sklearn PCA): The fitted PCA model.
        centroid (np.array): The [x, y] coordinates of the training data centroid.
        threshold (float): The maximum allowed distance (95th percentile).

    Returns:
        is_in_domain (bool): True of inside boundary, False if outside boundary.
        distance (float): The actual distance from the centroid.
    """
    fp_2d = fingerprint.reshape(1, -1)  # reshape the 1D array to 2D array
    fp_pca = pca_model.transform(fp_2d)
    distance = np.linalg.norm(fp_pca - centroid, axis=1)[0]
    is_in_domain = bool(distance <= threshold)

    return distance, is_in_domain
