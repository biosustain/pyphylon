"""
Functions for handling dimension-reduction models of pangenome data.
"""

import logging
from traceback import format_exc
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from prince import MCA
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics import confusion_matrix

from pyphylon.util import _get_normalization_diagonals

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

################################
#           Functions          #
################################


# Multiple Corresspondence Analysis (MCA)
def run_mca(data):
    """
    Run Multiple Correspondence Analysis (MCA) on the dataset.

    Parameters:
    - data: DataFrame containing the dataset to be analyzed.

    Returns:
    - MCA fitted model.
    """
    mca = MCA(n_components=min(data.shape), n_iter=1, copy=True, check_input=True, engine="sklearn", random_state=42)
    return mca.fit(data)


# Non-negative Matrix Factorization (NMF)
def run_nmf(data: Union[np.ndarray, pd.DataFrame], ranks: List[int], max_iter: int = 10_000):
    """
    Run NMF on the input data across multiple ranks.
    NMF decomposes a non-negative matrix D into two non-negative matrices W and H:
    D ≈ W * H

    Where:
    - D is the input data matrix (n_samples, n_features)
    - W is the basis matrix (n_samples, n_components)
    - H is the coefficient matrix (n_components, n_features)

    The optimization problem solved is:
    min_{W,H} 0.5 * ||D - WH||_Fro^2
    subject to W,H >= 0

    Non-negative Double Singlular Value Decomposition (NNDSVD) is used
    for the intialization of the optimization problem. This is done to
    ensure the basis matrix (and correspondingly the coefficient matrix)
    is as sparse as possible.

    Parameters:
    - data: DataFrame containing the dataset to be analyzed.
    - ranks: List of ranks (components) to try.
    - max_iter: Maximum number of iterations to try to reach convergence.

    Returns:
    - W_dict: A dictionary of transformed data at various ranks.
    - H_dict: A dictionary of model components at various ranks.

    Notes:
    ------
    This function uses the 'nndsvd' initialization, which is based on two SVD processes,
    one approximating the data matrix, the other approximating positive sections of
    the resulting partial SVD factors.

    References:
    -----------
    Boutsidis, C., & Gallopoulos, E. (2008). SVD based initialization: A head start
    for nonnegative matrix factorization. Pattern Recognition, 41(4), 1350-1362.
    """
    # Data validation
    if data.ndim != 2:
        raise ValueError("data must be a 2-dimensional array")
    if not all(r > 0 for r in ranks):
        raise ValueError("ranks must be a list of positive integers")
    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    # Initialize outputs
    W_dict, H_dict = {}, {}

    # Run NMF at varying ranks
    logger.info(f"Starting NMF process for {len(ranks)} ranks")
    for rank in ranks:
        model = NMF(n_components=rank, init="nndsvd", max_iter=max_iter, random_state=42)

        logger.info(f"Fitting NMF model for rank {rank}")
        W = model.fit_transform(data)
        H = model.components_
        W_dict[rank] = W
        H_dict[rank] = H

    logger.info("NMF process completed for given ranks")
    return W_dict, H_dict


def normalize_nmf_outputs(
    data: pd.DataFrame, W_dict: Dict[int, np.ndarray], H_dict: Dict[int, np.ndarray]
) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Normalize NMF outputs (99th percentile of W, column-by-column).

    Parameters:
    - data: Original dataset used for NMF.
    - W_dict: Dictionary containing W matrices.
    - H_dict: Dictionary containing H matrices.

    Returns:
    - L_norm_dict: Normalized L matrices.
    - A_norm_dict: Normalized A matrices.
    """
    L_norm_dict, A_norm_dict = {}, {}
    for rank, W in W_dict.items():
        try:
            H = H_dict[rank]
            D1, D2 = _get_normalization_diagonals(pd.DataFrame(W))
            L_norm_dict[rank] = pd.DataFrame(np.dot(W, D1), index=data.index)
            A_norm_dict[rank] = pd.DataFrame(np.dot(D2, H), columns=data.columns)
        except KeyError:
            logging.warning(f"Rank {rank} not found in H_dict. Skipping...")  # TODO: update to long-form logging
        except ValueError as e:
            logging.error(f"Error normalizing matrices for rank {rank}:\n{format_exc()}\nThe exception above occured for rank {rank} and will be ignored")

    return L_norm_dict, A_norm_dict


def binarize_nmf_outputs(L_norm_dict, A_norm_dict):
    """
    Binarize NMF outputs using k-means clustering (k=3, top cluster only).

    Parameters:
    - L_norm_dict: Dictionary of normalized L matrices.
    - A_norm_dict: Dictionary of normalized A matrices.

    Returns:
    - L_binarized_dict: Binarized L matrices.
    - A_binarized_dict: Binarized A matrices.
    """
    L_binarized_dict, A_binarized_dict = {}, {}
    for rank in L_norm_dict:
        try:
            L_binarized_dict[rank] = _k_means_binarize_L(L_norm_dict[rank])
            A_binarized_dict[rank] = _k_means_binarize_A(A_norm_dict[rank])
        except ValueError:
            logging.error(f"Error binarizing matrices for rank {rank}:\n{format_exc()}\nThe exception above occured for rank {rank} and will be ignored")
    return L_binarized_dict, A_binarized_dict


def generate_nmf_reconstructions(data, L_binarized_dict, A_binarized_dict):
    """
    Calculate model reconstr, error, & confusion matrix for each L_bin & A_bin
    """
    P_reconstructed_dict = {}
    P_error_dict = {}
    P_confusion_dict = {}

    for rank in L_binarized_dict:
        reconstr, err, confusion = _calculate_nmf_reconstruction(data, L_binarized_dict[rank], A_binarized_dict[rank])

        P_reconstructed_dict[rank] = reconstr
        P_error_dict[rank] = err
        P_confusion_dict[rank] = confusion

    return P_reconstructed_dict, P_error_dict, P_confusion_dict


# Calculate model reconstruction metrics
def calculate_nmf_reconstruction_metrics(P_reconstructed_dict, P_confusion_dict):
    """
    Calculate all reconstruction metrics from the generated confusion matrix
    """
    df_metrics = pd.DataFrame()

    for rank in P_reconstructed_dict:
        df_metrics[rank] = _calculate_metrics(P_confusion_dict[rank], P_reconstructed_dict[rank], rank)

    df_metrics = df_metrics.T
    df_metrics.index.name = "rank"

    return df_metrics


# Helper functions
def _k_means_binarize_L(L_norm):
    """
    Use k-means clustering (k=3) to binarize L_norm matrix.
    """

    # Initialize an empty array to hold the binarized matrix
    L_binarized = np.zeros_like(L_norm.values)

    # Loop through each column
    for col_idx in range(L_norm.values.shape[1]):
        column_data = L_norm.values[:, col_idx]

        # Reshape the column data to fit the KMeans input shape
        column_data_reshaped = column_data.reshape(-1, 1)

        # Apply 3-means clustering (gen better P-R tradeoff than 2-means)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
        kmeans.fit(column_data_reshaped)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Find the cluster with the highest mean
        highest_mean_cluster = np.argmax(centers)

        # Binarize the column based on the cluster with the highest mean
        binarized_column = (labels == highest_mean_cluster).astype(int)

        # Update the binarized matrix
        L_binarized[:, col_idx] = binarized_column

    # Typecast to DataFrame
    L_binarized = pd.DataFrame(L_binarized, index=L_norm.index, columns=L_norm.columns)
    return L_binarized


def _k_means_binarize_A(A_norm):
    """
    Use k-means clustering (k=3) to binarize A_norm matrix.
    """
    # Initialize an empty array to hold the binarized matrix
    A_binarized = np.zeros_like(A_norm.values)

    # Loop through each row
    for row_idx in range(A_norm.values.shape[0]):
        row_data = A_norm.values[row_idx, :]

        # Reshape the row data to fit the KMeans input shape
        row_data_reshaped = row_data.reshape(-1, 1)

        # Apply 3-means clustering (gen better P-R tradeoff than 2-means)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
        kmeans.fit(row_data_reshaped)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Find the cluster with the highest mean
        highest_mean_cluster = np.argmax(centers)

        # Binarize the row based on the cluster with the highest mean
        binarized_row = (labels == highest_mean_cluster).astype(int)

        # Update the binarized matrix
        A_binarized[row_idx, :] = binarized_row

    # Typecast to DataFrame
    A_binarized = pd.DataFrame(A_binarized, index=A_norm.index, columns=A_norm.columns)
    return A_binarized


def _calculate_nmf_reconstruction(data, L_binarized, A_binarized):

    # Multiply the binarized matrices to get the reconstructed matrix
    P_reconstructed = pd.DataFrame(np.dot(L_binarized, A_binarized), index=data.index, columns=data.columns)

    # Calculate the error matrix
    P_error = data - P_reconstructed

    # Binarize the orig and reconstr matrices for confusion matrix calculation
    data_binary = (data.values > 0).astype("int8")
    P_reconstructed_binary = (P_reconstructed.values > 0).astype("int8")

    # Flatten the matrices to use them in the confusion matrix calculation
    data_flat = data_binary.flatten()
    P_reconstructed_flat = P_reconstructed_binary.flatten()

    # Generate the confusion matrix
    # Definitions:
    # True Positive (TP): both actual and predicted are true
    # False Positive (FP): actual is false, but predicted is true
    # True Negative (TN): both actual and predicted are false
    # False Negative (FN): actual is true, but predicted is false
    P_confusion = confusion_matrix(data_flat, P_reconstructed_flat, labels=[1, 0])

    return P_reconstructed, P_error, P_confusion


def _calculate_metrics(P_confusion, P_reconstructed, rank):

    # Unpack confusion matrix elements
    TP = P_confusion[0, 0]
    FN = P_confusion[0, 1]
    FP = P_confusion[1, 0]
    TN = P_confusion[1, 1]

    # Use float for calculations to prevent integer overflow
    TP, FN, FP, TN = map(float, [TP, FN, FP, TN])
    Total = TP + TN + FP + FN

    # Calculations
    Precision = TP / (TP + FP) if TP + FP != 0 else 0
    Recall = TP / (TP + FN) if TP + FN != 0 else 0
    P_plus_R = Precision + Recall
    FPR = FP / (FP + TN) if FP + TN != 0 else 0
    FNR = FN / (TP + FN) if TP + FN != 0 else 0
    Specificity = TN / (TN + FP) if TN + FP != 0 else 0
    Prevalence = (TP + FN) / Total
    Accuracy = (TP + TN) / Total
    F1_score = 2 * (Precision * Recall) / (P_plus_R) if P_plus_R != 0 else 0
    BM = Recall + Specificity - 1  # a.k.a Youden's J statistic
    Jaccard_index = TP / (TP + FP + FN) if TP + FP + FN != 0 else 0

    # Adjusted MCC calculation to avoid overflow
    numerator = TP * TN - FP * FN
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = numerator / denominator if denominator != 0 else 0

    # Adjusted Prevalence Threshold to avoid overflow
    one_minus_Sp = 1 - Specificity
    if BM != 0:
        PT = (np.sqrt(Recall * one_minus_Sp) + Specificity - 1) / (BM)
    else:
        PT = 0

    # Calculate Akaike Information Criterion (AIC)
    Reconstruction_error = 1 - Jaccard_index  # Jaccard distance (proxy for reconstr error)
    k = 2 * rank * (P_reconstructed.shape[0] + P_reconstructed.shape[1])  # number of parameters in NMF (W & H matrices)
    AIC = 2 * k + 2 * Reconstruction_error * Total

    return {
        "Precision": Precision,
        "Recall": Recall,
        "FPR": FPR,
        "FNR": FNR,
        "Specificity": Specificity,
        "Prevalence": Prevalence,
        "Accuracy": Accuracy,
        "F1 Score": F1_score,
        "BM": BM,
        "Prevalence Threshold": PT,
        "MCC": MCC,
        "Jaccard Index": Jaccard_index,
        "AIC": AIC,
    }


def recommended_threshold(A_norm, i):
    column_data_reshaped = A_norm.loc[f"phylon{i}"].values.reshape(-1, 1)

    # 3-means clustering
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
    kmeans.fit(column_data_reshaped)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Find the cluster with the highest mean
    highest_mean_cluster = np.argmax(centers)

    # Binarize the row based on the cluster with the highest mean
    binarized_row = (labels == highest_mean_cluster).astype(int)

    # Find k-means-recommended threshold using min value that still binarizes to 1
    x = pd.Series(dict(zip(A_norm.columns, binarized_row)))
    threshold = A_norm.loc[f"phylon{i}", x[x == 1].index].min()

    return threshold
