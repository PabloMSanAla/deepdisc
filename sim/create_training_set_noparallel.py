# %%
##  This script helps create a training set for machine learning models by
##  simulating surface brightness profiles with various parameters.  ###


import os
from collections.abc import Iterable

# At the beginning of the file, add this import if not present
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from scipy import ndimage
from scipy.optimize import fsolve
from scipy.special import gammaincinv
from tqdm import tqdm


def disc_weights(r, rbreaks, beta):
    """
    Function to generate the softening function for the breaks in the disc

    Parameters
    ----------
    r : array-like
        The radii at which the softening function will be computed
    rbreaks : array-like
        The radii at which the scale length changes
    beta : float
        The softening parameter

    Returns
    -------
        Wb : array-like
            The weights for each disc components
    """
    rbreaks = np.array(rbreaks)
    r = np.array(r)
    Wb = (1 / np.pi) * (
        np.pi / 2 + np.arctan2(r[:, np.newaxis] - rbreaks[np.newaxis, :], beta)
    )
    return Wb


def multiple_exponential_discs(r, mu0, h, rbreaks, modelsOut=False):
    """
    Function to generate multiple exponentials discs
    in surface brightness units (mag arcsec^-2)

    Parameters
    ----------
    r : array-like
        The radii at which the surface brightness will be computed
    mu0 : float
        The central surface brightness of the first component disc
    h : array-like
        The scale lengths of the disc components
    rbreaks : array-like
        The radii at which the scale length changes
    modelsOut : bool, optional
        If True, the function will return the surface brightness of each disc
        component as well as the total surface brightness. Default is False.

    Returns
    -------
        mu : array-like
            The surface brightness profile

    """

    # Make sure that the inputs are numpy arrays
    r = np.array(r)
    h = np.array(h)
    rbreaks = np.array(rbreaks)

    # Check how many discs, and raise error if incompatible
    components = len(h)
    number_breaks = len(rbreaks)
    if components - number_breaks != 1:
        raise ValueError(
            f"The number of scale lengths ({components}) must be one more than the number of breaks ({number_breaks}) "
        )

    # Check that the number of components is at least 2
    if components <= 1:
        raise ValueError("At least two disc component must be provided")

    # Generate weights maps for each disc component
    Wb = disc_weights(r, rbreaks, 1e-6)
    Wb = np.append(
        np.ones_like(r)[:, np.newaxis],
        np.append(Wb, np.zeros_like(r)[:, np.newaxis], axis=1),
        axis=1,
    )

    # Compute the mu0 for each component of the disc
    a = 2.5 * (1 / np.log(10))
    factor = a * rbreaks * (h[1:] - h[:-1]) / (h[:-1] * h[1:])
    mu0s = np.append(mu0, mu0 * np.ones_like(factor) + np.cumsum(factor))

    # Compute the surface brightness profile
    mu_pre = mu0s[np.newaxis, :] + a * r[:, np.newaxis] / h[np.newaxis, :]
    mu = np.sum(mu_pre * Wb[:, :-1] * (1 - Wb[:, 1:]), axis=1)
    output = (mu, mu_pre) if modelsOut else mu
    return output


def exponential_disc_sb(r, mu_0, h):
    """Exponential disc in a surface brightness profile

    Parameters:
    -----------
        r : float
            Radius in arcsec
        mu_0 : float
            Central surface brightness in mag/arcsec^2
        h : float
            Scale length in arcsec

    Returns:
    --------
        mu : float
            Surface brightness at radius r"""
    a = 2.5 * (1 / np.log(10))
    return mu_0 + a * r * (1 / h)


def sersic_sb(r, mu_e, r_e, n):
    """Sersic profile in surface brightness units

    Parameters:
    -----------
        r : float
            Radius in arcsec
        mu_e : float
            Surface brightness at r_e
        r_e : float
            Effective radius in arcsec
        n : float
            Sersic index

    Returns:
    --------
        mu : float
            Surface brightness at radius r"""
    bn = gammaincinv(2 * n, 0.5)
    return mu_e + 2.5 * (1 / np.log(10)) * bn * ((r / r_e) ** (1 / n) - 1)


def sersic_sb_truncated(r, mu_e, r_e, n, r_trunc, steepness=10):
    """Truncated Sersic profile in surface brightness units

    Parameters:
    -----------
        r : float
            Radius in arcsec
        mu_e : float
            Surface brightness at r_e
        r_e : float
            Effective radius in arcsec
        n : float
            Sersic index
        r_trunc : float
            Truncation radius in arcsec
        steepness : float, optional
            Steepness of the truncation. Default is 10.

    Returns:
    --------
        mu : float
            Surface brightness at radius r"""

    mu_sersic = sersic_sb(r, mu_e, r_e, n)
    truncation = normalized_sigmoid(r, x0=r_trunc, steepness=steepness)
    return mu_sersic - 30 * (truncation - 1)


def multicomponent_model(r, mue, re, n, mu0, h, rbreaks):
    """Function to generate a multi-component model
    with a bulge modeled by a Sersic and multiple exponential discs

    Parameters:
    -----------

        r : array-like
            The radii at which the surface brightness will be computed
        mue : float
            Surface brightness at the effective radius
        re : float
            Effective radius in units of r
        n : float
            Sersic index
        mu0 : float
            Central surface brightness of the first disc component
        h : float or array-like
            The scale lengths of the disc components
        rbreaks : float or array-like
            The radii at which the scale length changes

    Returns:
    --------
        mu : array-like
            The surface brightness
    """
    # Define the different components of the model
    if isinstance(h, Iterable) and len(h) > 1:
        disc_comp = multiple_exponential_discs(r, mu0, h, rbreaks)
    else:
        disc_comp = exponential_disc_sb(r, mu0, h[0])

    # Compute the bulge surface brightness
    bulge = sersic_sb(r, mue, re, n)

    # Find the indices where the difference changes sign
    diff = disc_comp - bulge
    cross_indices = np.where(np.diff(np.sign(diff)))[0]

    # Calculate the crossing points using linear interpolation
    x1, x2 = r[cross_indices], r[cross_indices + 1]
    y1, y2 = diff[cross_indices], diff[cross_indices + 1]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    cross_points = -b / m

    # Define the radius of the bulge
    rbulge = np.nanmin(cross_points) if len(cross_points) > 0 else np.nanmax(r)
    bulge = sersic_sb_truncated(r, mue, re, n, rbulge)

    # Combine the components by adding them in linear normalized_sigmoid(r,x0=2*rbulge)
    total_linear = 10 ** (-0.4 * bulge) + 10 ** (-0.4 * disc_comp)
    mu = -2.5 * np.log10(total_linear)

    return mu


def check_constraint(mu_0, h, r_constraint, mu_e, r_e, n, num_points=1000):
    """
    Check if the exponential and Sersic profiles have a crossing point
    at a radius lower than r_constraint. Optimized version.

    Parameters:
    -----------
    mu_0 : float
        Central surface brightness of exponential disk (mag/arcsec^2)
    h : float
        Scale length of exponential disk (arcsec)
    r_constraint : float
        Maximum radius to check for crossing (arcsec)
    mu_e : float
        Surface brightness at effective radius for Sersic profile (mag/arcsec^2)
    r_e : float
        Effective radius of Sersic profile (arcsec)
    n : float
        Sersic index
    num_points : int
        Number of points to sample for finding crossing

    Returns:
    --------
    has_crossing : bool
        True if profiles cross at r < r_constraint
    crossing_radius : float or None
        Radius where profiles cross, or None if no crossing found
    """
    # Sample the difference using vectorized operations
    r_sample = np.linspace(0.01, r_constraint * 1.5, num_points)

    # Vectorized computation of both profiles
    exp_sb = exponential_disc_sb(r_sample, mu_0, h)
    sersic_profile = sersic_sb(r_sample, mu_e, r_e, n)
    diff_values = exp_sb - sersic_profile

    # Find sign changes (potential crossing points)
    sign_changes = np.where(np.diff(np.sign(diff_values)))[0]

    if len(sign_changes) == 0:
        return False, None

    # Check only crossings within r_constraint
    for idx in sign_changes:
        r_left = r_sample[idx]
        r_right = r_sample[idx + 1]

        # Linear interpolation for quick crossing estimate
        y_left = diff_values[idx]
        y_right = diff_values[idx + 1]
        r_cross = r_left - y_left * (r_right - r_left) / (y_right - y_left)

        if r_cross < r_constraint and r_cross > 0:
            return True, r_cross

    return False, None


def find_valid_sersic_params(
    mu_0,
    h,
    r_constraint,
    mu_e,
    n_range=(0.5, 8),
    r_e_range=None,
    n_points=20,
    r_e_points=20,
):
    """
    Find valid Sersic parameters (n, r_e) that have a crossing point with
    the exponential disk profile at r < r_constraint. Optimized version.

    Parameters:
    -----------
    mu_0 : float
        Central surface brightness of exponential disk (mag/arcsec^2)
    h : float
        Scale length of exponential disk (arcsec)
    r_constraint : float
        Maximum radius for crossing point
    mu_e : float
        Fixed surface brightness at effective radius for Sersic profile
    n_range : tuple
        Range of Sersic indices to search (min, max)
    r_e_range : tuple or None
        Range of effective radii to search. If None, uses (0.1*r_constraint, 10*r_constraint)
    n_points : int
        Number of points to sample in n dimension
    r_e_points : int
        Number of points to sample in r_e dimension

    Returns:
    --------
    valid_params : list of tuples
        List of (n, r_e, crossing_radius) tuples that satisfy the constraint
    grid_results : dict
        Dictionary with grid search results for plotting
    """
    if r_e_range is None:
        r_e_range = (0.1 * r_constraint, 10 * r_constraint)

    n_grid = np.linspace(n_range[0], n_range[1], n_points)
    r_e_grid = np.linspace(r_e_range[0], r_e_range[1], r_e_points)

    # Pre-allocate arrays
    constraint_satisfied = np.zeros((n_points, r_e_points), dtype=bool)
    crossing_radii = np.full((n_points, r_e_points), np.nan)

    # Vectorized sampling grid for all profiles
    r_sample = np.linspace(0.01, r_constraint * 1.5, 100)
    exp_sb = exponential_disc_sb(r_sample, mu_0, h)

    valid_params = []

    # Use vectorization where possible
    for i, n in enumerate(n_grid):
        bn = gammaincinv(2 * n, 0.5)  # Compute once per n

        for j, r_e in enumerate(r_e_grid):
            # Vectorized Sersic computation
            sersic_sb_vals = mu_e + 2.5 * (1 / np.log(10)) * bn * (
                (r_sample / r_e) ** (1 / n) - 1
            )
            diff_values = exp_sb - sersic_sb_vals

            # Find sign changes
            sign_changes = np.where(np.diff(np.sign(diff_values)))[0]

            if len(sign_changes) > 0:
                # Check first crossing only
                idx = sign_changes[0]
                r_left = r_sample[idx]
                r_right = r_sample[idx + 1]

                # Linear interpolation
                y_left = diff_values[idx]
                y_right = diff_values[idx + 1]
                r_cross = r_left - y_left * (r_right - r_left) / (
                    y_right - y_left
                )

                if r_cross < r_constraint and r_cross > 0:
                    constraint_satisfied[i, j] = True
                    crossing_radii[i, j] = r_cross
                    valid_params.append((r_e, n, r_cross))

                # Add that the slope should be similar at the crossing point 
                # to ensure a smooth transition (optional, can be removed if not needed)

                

    grid_results = {
        "n_grid": n_grid,
        "r_e_grid": r_e_grid,
        "constraint_satisfied": constraint_satisfied,
        "crossing_radii": crossing_radii,
    }

    return valid_params, grid_results


def sample_sersic_params_stratified(
    valid_params, n_samples, n_bins=5, r_cross_bins=5
):
    """
    Sample Sersic parameters (n, r_e) homogeneously across both n and crossing radius bins.
    Uses 2D stratified sampling for maximum homogeneity.

    Parameters:
    -----------
    valid_params : list of tuples
        List of (n, r_e, crossing_radius) tuples from find_valid_sersic_params
    n_samples : int
        Total number of samples to draw
    n_bins : int
        Number of bins to stratify by Sersic index n
    r_cross_bins : int
        Number of bins to stratify by crossing radius

    Returns:
    --------
    sampled_params : list of tuples
        Sampled (n, r_e, crossing_radius) tuples, distributed homogeneously in 2D
    """
    if len(valid_params) == 0:
        return []

    # Convert to numpy array for easier manipulation
    params_array = np.array(valid_params)

    # Get n values (column 0) and crossing radii (column 2)
    n_values = params_array[:, 0]
    crossing_radii = params_array[:, 2]

    # Create 2D bins
    n_bin_edges = np.linspace(n_values.min(), n_values.max(), n_bins + 1)
    r_bin_edges = np.linspace(
        crossing_radii.min(), crossing_radii.max(), r_cross_bins + 1
    )

    # Assign each parameter set to a 2D bin
    n_bin_indices = np.digitize(n_values, n_bin_edges[:-1]) - 1
    n_bin_indices = np.clip(n_bin_indices, 0, n_bins - 1)

    r_bin_indices = np.digitize(crossing_radii, r_bin_edges[:-1]) - 1
    r_bin_indices = np.clip(r_bin_indices, 0, r_cross_bins - 1)

    # Count total number of 2D bins
    total_bins = n_bins * r_cross_bins
    samples_per_bin = n_samples // total_bins
    remaining_samples = n_samples % total_bins

    sampled_params = []
    bin_counter = 0

    # Iterate over all 2D bins
    for n_idx in range(n_bins):
        for r_idx in range(r_cross_bins):
            # Get all params in this 2D bin
            bin_mask = (n_bin_indices == n_idx) & (r_bin_indices == r_idx)
            bin_params = params_array[bin_mask]

            if len(bin_params) == 0:
                bin_counter += 1
                continue

            # Number of samples for this bin
            n_bin_samples = samples_per_bin + (
                1 if bin_counter < remaining_samples else 0
            )

            # Sample with replacement if needed
            if n_bin_samples > 0:
                sample_indices = np.random.choice(
                    len(bin_params), size=n_bin_samples, replace=True
                )
                sampled_params.extend(
                    [tuple(bin_params[i]) for i in sample_indices]
                )

            bin_counter += 1

    return sampled_params


def sample_columns_stratified(
    data,
    columns_to_sample,
    stratify_column,
    n_samples,
    n_bins=5,
    stratify_column_2=None,
    n_bins_2=5,
):
    """
    Sample specific columns homogeneously based on 1D or 2D stratification.

    Parameters:
    -----------
    data : array-like or DataFrame
        The data to sample from (N x M array or DataFrame)
    columns_to_sample : list of int or str
        Column indices (for arrays) or names (for DataFrames) to sample
    stratify_column : int or str
        Column index (for arrays) or name (for DataFrame) to stratify by
    n_samples : int
        Total number of samples to draw
    n_bins : int
        Number of bins for first stratification dimension
    stratify_column_2 : int or str, optional
        Second column to stratify by for 2D stratification (for homogeneity in 2 dimensions)
    n_bins_2 : int
        Number of bins for second stratification dimension

    Returns:
    --------
    sampled_data : ndarray or DataFrame
        Sampled data with homogeneous distribution across stratify_column(s) bins

    Examples:
    ---------
    # For numpy array: sample columns 0 and 1, stratified by column 2
    sampled = sample_columns_stratified(data, [0, 1], 2, n_samples=100, n_bins=5)

    # For 2D stratification: homogeneous in both column 0 and column 2
    sampled = sample_columns_stratified(data, [0, 1], 0, n_samples=100, n_bins=5,
                                       stratify_column_2=2, n_bins_2=5)

    # For DataFrame
    sampled = sample_columns_stratified(df, ['n', 'r_e'], 'n', n_samples=100,
                                       stratify_column_2='r_cross')
    """
    # Handle both numpy arrays and pandas DataFrames
    if isinstance(data, pd.DataFrame):
        stratify_values = data[stratify_column].values
        is_dataframe = True
    else:
        data = np.array(data)
        stratify_values = data[:, stratify_column]
        is_dataframe = False

    # 2D stratification if second column provided
    if stratify_column_2 is not None:
        if is_dataframe:
            stratify_values_2 = data[stratify_column_2].values
        else:
            stratify_values_2 = data[:, stratify_column_2]

        # Create 2D bins
        bin_edges_1 = np.linspace(
            stratify_values.min(), stratify_values.max(), n_bins + 1
        )
        bin_edges_2 = np.linspace(
            stratify_values_2.min(), stratify_values_2.max(), n_bins_2 + 1
        )

        # Assign to 2D bins
        bin_indices_1 = np.digitize(stratify_values, bin_edges_1[:-1]) - 1
        bin_indices_1 = np.clip(bin_indices_1, 0, n_bins - 1)

        bin_indices_2 = np.digitize(stratify_values_2, bin_edges_2[:-1]) - 1
        bin_indices_2 = np.clip(bin_indices_2, 0, n_bins_2 - 1)

        # Build list of non-empty 2D bins
        non_empty_bins = []
        for bin_idx_1 in range(n_bins):
            for bin_idx_2 in range(n_bins_2):
                bin_mask = (bin_indices_1 == bin_idx_1) & (
                    bin_indices_2 == bin_idx_2
                )
                bin_row_indices = np.where(bin_mask)[0]
                if len(bin_row_indices) > 0:
                    non_empty_bins.append(bin_row_indices)

        sampled_indices = []
        if len(non_empty_bins) > 0:
            samples_per_bin = n_samples // len(non_empty_bins)
            remaining_samples = n_samples % len(non_empty_bins)

            for i, bin_row_indices in enumerate(non_empty_bins):
                n_bin_samples = samples_per_bin + (
                    1 if i < remaining_samples else 0
                )

                if n_bin_samples > 0:
                    selected = np.random.choice(
                        bin_row_indices, size=n_bin_samples, replace=True
                    )
                    sampled_indices.extend(selected)

    else:
        # 1D stratification (original behavior)
        bin_edges = np.linspace(
            stratify_values.min(), stratify_values.max(), n_bins + 1
        )
        bin_indices = np.digitize(stratify_values, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        non_empty_bins = []
        for bin_idx in range(n_bins):
            bin_mask = bin_indices == bin_idx
            bin_row_indices = np.where(bin_mask)[0]
            if len(bin_row_indices) > 0:
                non_empty_bins.append(bin_row_indices)

        sampled_indices = []
        if len(non_empty_bins) > 0:
            samples_per_bin = n_samples // len(non_empty_bins)
            remaining_samples = n_samples % len(non_empty_bins)

            for i, bin_row_indices in enumerate(non_empty_bins):
                n_bin_samples = samples_per_bin + (
                    1 if i < remaining_samples else 0
                )

                if n_bin_samples > 0:
                    selected = np.random.choice(
                        bin_row_indices, size=n_bin_samples, replace=True
                    )
                    sampled_indices.extend(selected)

    # Ensure we always return exactly n_samples
    total_rows = len(data)
    if len(sampled_indices) < n_samples:
        n_needed = n_samples - len(sampled_indices)
        all_indices = np.arange(total_rows)
        if len(sampled_indices) == 0:
            remaining_indices = all_indices
        else:
            remaining_indices = np.setdiff1d(all_indices, sampled_indices)

        if len(remaining_indices) >= n_needed:
            fill_indices = np.random.choice(
                remaining_indices, size=n_needed, replace=False
            )
        else:
            fill_indices = list(remaining_indices)
            extra_needed = n_needed - len(remaining_indices)
            if extra_needed > 0:
                fill_indices = np.concatenate(
                    [
                        np.array(fill_indices, dtype=int),
                        np.random.choice(
                            all_indices, size=extra_needed, replace=True
                        ),
                    ]
                )
        sampled_indices.extend(fill_indices.tolist())
    elif len(sampled_indices) > n_samples:
        sampled_indices = sampled_indices[:n_samples]

    # Return sampled data
    if is_dataframe:
        if isinstance(columns_to_sample, list):
            # Avoid duplicating stratification columns if already in columns_to_sample
            cols_to_return = list(columns_to_sample)
            if stratify_column not in cols_to_return:
                cols_to_return.append(stratify_column)
            if (
                stratify_column_2 is not None
                and stratify_column_2 not in cols_to_return
            ):
                cols_to_return.append(stratify_column_2)
            return data.iloc[sampled_indices][cols_to_return].reset_index(
                drop=True
            )
        else:
            return data.iloc[sampled_indices].reset_index(drop=True)
    else:
        if isinstance(columns_to_sample, list):
            # Avoid duplicating stratification columns if already in columns_to_sample
            all_cols = list(columns_to_sample)
            if stratify_column not in all_cols:
                all_cols.append(stratify_column)
            if (
                stratify_column_2 is not None
                and stratify_column_2 not in all_cols
            ):
                all_cols.append(stratify_column_2)
            return data[sampled_indices][:, all_cols]
        else:
            return data[sampled_indices]


def find_crossing_points(r, bulge, disc_comp):
    """Find the crossing points between bulge and disc components

    Parameters:
    -----------
        r : array-like
            The radii at which the surface brightness is computed
        bulge : array-like
            The bulge surface brightness profile
        disc_comp : array-like
            The disc surface brightness profile

    Returns:
    --------
        cross_points : array-like
            The radii where the bulge and disc components cross
    """
    diff = disc_comp - bulge
    cross_indices = np.where(np.diff(np.sign(diff)))[0]

    # Calculate the crossing points using linear interpolation
    x1, x2 = r[cross_indices], r[cross_indices + 1]
    y1, y2 = diff[cross_indices], diff[cross_indices + 1]
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    cross_points = -b / m

    return cross_points


def normalized_sigmoid(x, x0=0, steepness=10):
    """
    Normalized sigmoid function with adjustable steepness.

    Parameters:
    -----------
    x : float or array-like
        Input value(s).
    x0 : float
        Center point of the sigmoid transition.
    steepness : float
        Controls the steepness of the transition. Higher values create
        a sharper drop. Default is 10 for a fast transition near crossing points.

    Returns:
    --------
    float or array-like
        Normalized sigmoid output between 0 and 1.
    """
    return 1 - 1 / (1 + np.exp(-steepness * (x - x0)))

def normalized_sigmoid(x, x0=0, steepness=10, decay_start=None, decay_slope=0.0):
    """
    Normalized sigmoid function with adjustable steepness and optional linear decay.

    Parameters:
    -----------
    x : float or array-like
        Input value(s).
    x0 : float
        Center point of the sigmoid transition.
    steepness : float
        Controls the steepness of the transition. Higher values create
        a sharper drop. Default is 10.
    decay_start : float or None
        If provided, a linear decay starts at this x. If None (default) no decay applied.
    decay_slope : float
        Slope of the linear decay applied for x > decay_start. Default 0.0 (no decay).

    Returns:
    --------
    float or array-like
        Normalized sigmoid output between 0 and 1 (clipped after applying decay).
    """
    x_arr = np.asarray(x)
    sig = 1 - 1 / (1 + np.exp(-steepness * (x_arr - x0)))

    if decay_start is None or decay_slope == 0.0:
        out = sig
    else:
        decay = np.where(x_arr > decay_start, (x_arr - decay_start) * decay_slope, 0.0)
        out = sig - decay
        out = np.clip(out, 0.0, 1.0)

    return out


def convolve_profile(profile, r, kernel, kernel_r=None, method="scipy"):
    """
    Convolve a 1D radial profile with a kernel (e.g., PSF)

    Parameters
    ----------
    profile : array-like
        The 1D surface brightness profile values
    r : array-like
        The radial coordinates corresponding to the profile
    kernel : array-like
        The convolution kernel (e.g., PSF profile)
    kernel_r : array-like, optional
        The radial coordinates for the kernel. If None, assumes same spacing as r
    method : str, optional
        Convolution method: 'scipy', 'numpy', or 'fft'. Default is 'scipy'

    Returns
    -------
    convolved_profile : array-like
        The convolved profile, same length as input profile

    Notes
    -----
    - The kernel should be normalized (sum to 1) for flux conservation
    - For astronomical PSF convolution, the kernel typically represents seeing
    - The function handles edge effects using 'reflect' mode
    """

    profile = np.array(profile)
    r = np.array(r)
    kernel = np.array(kernel)

    # Normalize kernel to conserve flux
    kernel_normalized = kernel / np.sum(kernel)

    # If kernel coordinates not provided, assume same spacing as profile
    if kernel_r is None:
        dr = np.median(np.diff(r))
        kernel_size = len(kernel)
        kernel_r = np.arange(-(kernel_size // 2), kernel_size // 2 + 1) * dr

    # Choose convolution method
    if method == "scipy":
        # Use scipy's ndimage convolution with reflection at boundaries
        convolved = ndimage.convolve1d(
            profile, kernel_normalized, mode="reflect"
        )

    elif method == "numpy":
        # Use numpy's convolution with proper centering
        convolved_full = np.convolve(profile, kernel_normalized, mode="full")
        # Extract the central part to match original profile length
        start = len(kernel_normalized) // 2
        end = start + len(profile)
        convolved = convolved_full[start:end]

    elif method == "fft":
        # Use FFT-based convolution for large kernels
        from scipy.signal import fftconvolve

        convolved_full = fftconvolve(profile, kernel_normalized, mode="full")
        start = len(kernel_normalized) // 2
        end = start + len(profile)
        convolved = convolved_full[start:end]

    else:
        raise ValueError("Method must be 'scipy', 'numpy', or 'fft'")

    return convolved


def gaussian_psf_kernel(fwhm, dr, r_max=None):
    """
    Generate a Gaussian PSF kernel for convolution

    Parameters
    ----------
    fwhm : float
        Full Width at Half Maximum of the PSF in same units as dr
    dr : float
        Radial spacing
    r_max : float, optional
        Maximum radius for kernel. If None, uses 3*fwhm

    Returns
    -------
    kernel : array-like
        Normalized Gaussian kernel
    kernel_r : array-like
        Radial coordinates for the kernel
    """

    if r_max is None:
        r_max = 3 * fwhm

    # Convert FWHM to sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Create radial grid for kernel
    n_points = int(2 * r_max / dr) + 1
    kernel_r = np.linspace(-r_max, r_max, n_points)

    # Generate Gaussian kernel
    kernel = np.exp(-0.5 * (kernel_r / sigma) ** 2)

    # Normalize
    kernel = kernel / np.sum(kernel)

    return kernel, kernel_r


# Example usage function that fits with your code structure
def add_seeing_to_profile(model, r, seeing_fwhm):
    """
    Add seeing effects to a surface brightness profile

    Parameters
    ----------
    model : MultiComponentModel or callable
        The model object or function that generates the profile
    r : array-like
        Radial coordinates
    seeing_fwhm : float
        Seeing FWHM in same units as r

    Returns
    -------
    convolved_profile : array-like
        Surface brightness profile with seeing effects
    """

    # Generate the intrinsic profile
    if hasattr(model, "__call__"):
        intrinsic_profile = model(r)
    else:
        intrinsic_profile = model

    # Convert to linear flux units for convolution
    linear_profile = 10 ** (-0.4 * intrinsic_profile)

    # Create PSF kernel
    dr = np.median(np.diff(r))
    psf_kernel, _ = gaussian_psf_kernel(seeing_fwhm, dr)

    # Convolve in linear space
    convolved_linear = convolve_profile(linear_profile, r, psf_kernel)

    # Convert back to magnitude space
    convolved_profile = -2.5 * np.log10(convolved_linear)

    return convolved_profile


class MultiComponentModel:
    """Class to generate a multi-component model
    with a bulge modeled by a Sersic and multiple exponential discs

    Attributes:
    -----------
        mue : float
            Surface brightness at the effective radius
        re : float
            Effective radius in units of r
        n : float
            Sersic index
        mu0 : float
            Central surface brightness of the first disc component
        h : float or array-like
            The scale lengths of the disc components
        rbreaks : float or array-like
            The radii at which the scale length changes

    Methods:
    --------
        __call__(r) : Generates the multi-component model

        function_to_fitter(r, *params) : Funtction to provide to curve_fit

    """

    def __init__(self, mue, re, n, mu0, h, rbreaks):
        """Initialize the model with the parameters

        Parameters:
        ----------
            mue : float
                Surface brightness at the effective radius
            re : float
                Effective radius in units of r
            n : float
                Sersic index
            mu0 : float
                Central surface brightness of the first disc component
            h : array-like
                The scale lengths of the disc components
            rbreaks : array-like
                The radii at which the scale length changes
        """
        self.mue = mue
        self.re = re
        self.n = n
        self.mu0 = mu0
        self.h = h
        self.rbreaks = rbreaks
        self.ndiscs = len(h)
        self.nbreaks = len(rbreaks)
        self.param_list = (
            [mue, re, n, mu0] + [x for x in h] + [x for x in rbreaks]
        )

    def __call__(self, r):
        """Call the model with the parameters"""
        return multicomponent_model(
            r, self.mue, self.re, self.n, self.mu0, self.h, self.rbreaks
        )

    def function_to_fitter(self, r, *params):
        """Function to provide to curve fit to fit the
        multi-component model to the data. The parameters
        are passed as a list, and the function returns the
        model surface brightness profile.

        The parameters are:
        [sersic_mu, sersic_r_e, sersic_n, mu0,
                h1, h2, ..., rbreaks1, rbreaks2, ...]

        Parameters:
        -----------
            r : array-like
                The radii at which the surface brightness will be computed
            params : list
                The parameters of the model. The first four are the Sersic parameters,
                and the rest are the disc parameters.
        Returns:
        --------
            mu : array-like
                The surface brightness profile
        """
        size = len(params)
        sersic_mu, sersic_re, sersic_n, mu0 = params[:4]
        d = size - 4
        ndiscs = 1 + d // 2
        h = params[4 : 4 + ndiscs]
        rbreaks = params[4 + ndiscs :]
        mu = multicomponent_model(
            r, sersic_mu, sersic_re, sersic_n, mu0, h, rbreaks
        )
        return mu

    def update_params(self, params):
        """Update the parameters of the model"""
        self.mue = params[0]
        self.re = params[1]
        self.n = params[2]
        self.mu0 = params[3]
        self.h = params[4 : 4 + self.ndiscs]
        self.rbreaks = params[4 + self.ndiscs :]
        self.param_list = (
            [self.mue, self.re, self.n, self.mu0]
            + [x for x in self.h]
            + [x for x in self.rbreaks]
        )

    def get_components(self, r):
        """Get the disc components of the model"""
        disc_comp = []
        mu0 = self.mu0
        for i in range(self.ndiscs):
            disc_comp += [exponential_disc_sb(r, mu0, self.h[i])]
            if i < len(self.rbreaks):
                mu0 = mu0 + (2.5 / np.log(10)) * self.rbreaks[i] * (
                    self.h[i + 1] - self.h[i]
                ) / (self.h[i] * self.h[i + 1])
        return disc_comp

    def get_dictionary(self, prefix=""):
        """Get the parameters of the model as a dictionary"""
        params = {
            f"{prefix}mue": self.mue,
            f"{prefix}re": self.re,
            f"{prefix}n": self.n,
            f"{prefix}mu0": self.mu0,
        }
        for i in range(self.ndiscs):
            params[f"{prefix}h{i+1}"] = self.h[i]
            if i < self.nbreaks:
                params[f"{prefix}rbreak{i+1}"] = self.rbreaks[i]
        params[f"{prefix}ndiscs"] = self.ndiscs
        return params

    def __str__(self):
        """String representation of the model"""
        params = self.get_dictionary()
        return f"Multi-component model with {self.ndiscs} discs:\n" + str(
            params
        )


# %%
if __name__ == "__main__":
    import os
    from pathlib import Path

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Set up paths
    data_dir = Path("data_test")
    obs_file = Path("observations/breaks_2025-04-15.csv")

    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Loading observational data...")
    # Load the observational data
    df = pd.read_csv(obs_file)

    # Filter valid break data (where breaks exist)
    valid_breaks = df[
        df["final_rbreaks1"].notna() & (df["final_rbreaks1"] > 0)
    ]
    print(f"Found {len(valid_breaks)} objects with valid breaks")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Number of simulated profiles to create
    n_profiles = 10

    # Radial grid for surface brightness profiles
    r_max = 50  # arcsec
    r = np.linspace(0.1, r_max, 200)

    # Sample parameters from the observational distribution
    print("Generating simulated profiles...")
    # %%

    # Define labels for break types
    # Invalid:  0
    # I:        1
    # II:       2
    # III:      3
    # II+III:   4
    # III+II:   5
    # II+II:    6
    # III+III:  7

    # Sersic:   0           @TODO

    valid_breaks = pd.read_csv(obs_file)
    valid_breaks = valid_breaks[valid_breaks["final_flag"]]
    valid_breaks["label"] = np.zeros_like(valid_breaks["final_rbreaks1"])
    valid_breaks.loc[(valid_breaks["final_ndiscs"] == 1), "label"] = 1
    valid_breaks.loc[
        (valid_breaks["final_ndiscs"] == 2)
        & (valid_breaks["final_h1"] > valid_breaks["final_h2"]),
        "label",
    ] = 2

    valid_breaks.loc[
        (valid_breaks["final_ndiscs"] == 2)
        & (valid_breaks["final_h1"] < valid_breaks["final_h2"]),
        "label",
    ] = 3

    valid_breaks.loc[
        (valid_breaks["final_ndiscs"] == 3)
        & (valid_breaks["final_h1"] > valid_breaks["final_h2"])
        & (valid_breaks["final_h2"] < valid_breaks["final_h3"]),
        "label",
    ] = 4
    valid_breaks.loc[
        (valid_breaks["final_ndiscs"] == 3)
        & (valid_breaks["final_h1"] < valid_breaks["final_h2"])
        & (valid_breaks["final_h2"] > valid_breaks["final_h3"]),
        "label",
    ] = 5

    valid_breaks.loc[
        (valid_breaks["final_ndiscs"] == 3)
        & (valid_breaks["final_h1"] > valid_breaks["final_h2"])
        & (valid_breaks["final_h2"] > valid_breaks["final_h3"]),
        "label",
    ] = 6

    valid_breaks.loc[
        (valid_breaks["final_ndiscs"] == 3)
        & (valid_breaks["final_h1"] < valid_breaks["final_h2"])
        & (valid_breaks["final_h2"] < valid_breaks["final_h3"]),
        "label",
    ] = 7

    # @TODO: Add Sersic profile identification
    break_types = np.arange(0, 8)
    percentages = {}
    for label in break_types:
        percentages[label] = np.sum(valid_breaks["label"] == label) / len(
            valid_breaks
        )

    N_samples = 200

    columns = {
        "final_mu0": 0.0,
        "final_h1": 0.0,
        "final_h2": 0.0,
        "final_h3": 0.0,
        "final_h4": 0.0,
        "final_ndiscs": 0,
        "final_rbreaks1": 0.0,
        "final_rbreaks2": 0.0,
        "final_rbreaks3": 0.0,
        "label": 0,
    }
    agmentation_params = {
        "augmentation_noise_level": 0.0,
        "augmentation_mue": 0.0,
        "augmentation_re": 0.0,
        "augmentation_n": 0.0,
        "augmentation_r_sersic": 0.0,
        "augmentation_deconvolution": False,
        "augmentation_shift_mu": 0.0,
        "augmentation_scale_r": 0.0,
        "augmentation_depth": 0.0,
        "augmentation_beta_break": 0.0,
    }

    ## Create Simulated Profiles

    # Set number of simulated profiles per break type
    lower_sample = 0.1 * N_samples
    N_simulated_per_type = {}
    for label in break_types:
        N_simulated_per_type[label] = lower_sample + int(
            np.round(percentages[label] * N_samples)
        )

    # Sample parameters for each profile
    types_array = np.concatenate(
        [[i] * int(N_simulated_per_type[i]) for i in break_types]
    )
    empty_columns = {col: [] for col in columns.keys()}
    simulated_params = pd.DataFrame(empty_columns)
    for lab in break_types:
        sample_pool = valid_breaks[valid_breaks["label"] == lab]
        n_samples = int(N_simulated_per_type[lab])
        sampled_table = sample_pool.sample(n=n_samples, replace=True)
        sampled_table["label"] = lab
        # Append to the simulated parameters DataFrame
        simulated_params = pd.concat(
            [simulated_params, sampled_table[columns.keys()]],
            ignore_index=True,
        )

    ## Augmentation of the parameters
    ### Sersic component        x4
    ### Noise component         x4
    ### PSF component           0
    ### Artifacts component     0. [edges at random region]
    ### ------------------------------
    ### Total augmentation      x16

    
    simulated_params_augmented = pd.DataFrame()
    _noise_std = np.array([0.1, 0.18, 0.25])
    N_sersic_samples = 3
    scale_augmentation = N_sersic_samples * _noise_std.size
    # Repeat each row N times sequentially
    simulated_params_augmented = simulated_params.loc[
        simulated_params.index.repeat(scale_augmentation)
    ].reset_index(drop=True)

    # Add augmentation parameter columns with default values
    for param, default_value in agmentation_params.items():
        simulated_params_augmented[param] = default_value

    images = []
    idx = 0
    failed = 0
    for i in tqdm(np.arange(len(simulated_params) - 1)):
        try:
            idx_previous = idx
            idx = (i + 1) * scale_augmentation
            # if i > 10:
            #     break

            # Get the main body of the exponential profile
            label = simulated_params_augmented.loc[idx - 1, "label"]

            if label == 1:
                mu0 = simulated_params_augmented.loc[idx - 1, "final_mu0"]
                h = simulated_params_augmented.loc[idx - 1, "final_h1"]
                r = np.logspace(-1, np.log10(12 * h), 200)
                mu = exponential_disc_sb(r, mu0, h)

            elif label == -1:  # TODO: Sersic profile
                continue
            else:
                rbreaks = simulated_params_augmented.loc[
                    idx - 1,
                    ["final_rbreaks1", "final_rbreaks2", "final_rbreaks3"],
                ].values
                h = simulated_params_augmented.loc[
                    idx - 1,
                    [
                        "final_h1",
                        "final_h2",
                        "final_h3",
                        "final_h4",
                    ],
                ].values
                mu0 = simulated_params_augmented.loc[idx - 1, "final_mu0"]
                r = np.logspace(-1, np.log10(20 * h[0]), 200)
                # remove zero from h and rbreaks
                h = h[h > 0]
                rbreaks = rbreaks[rbreaks > 0]
                mu = multiple_exponential_discs(
                    r, mu0, h.tolist(), rbreaks.tolist()
                )

            valid_params, _ = find_valid_sersic_params(
                mu_0=mu0,
                h=h[0] if isinstance(h, np.ndarray) else h,
                r_constraint=0.85*rbreaks[0],  # Your radius constraint
                mu_e=mu0,  # Fixed SB at effective radius
                n_range=(0.5, 4),  # Range of Sersic indices to search
                r_e_range=(
                    0.5,
                    5 * h[0] if isinstance(h, np.ndarray) else 5 * h,
                ),  # Range of effective radii to search,
                n_points=30,
                r_e_points=30,
            )

            _sampled_params = np.array(valid_params)[
                np.linspace(0, len(valid_params) - 1, N_sersic_samples).astype(np.int16), :
            ]

            final_param_sersic = np.column_stack(
                [
                    np.repeat(_sampled_params, _noise_std.size, axis=0),
                    np.tile(_noise_std, _sampled_params.shape[0]),
                ]
            )

            simulated_params_augmented.iloc[
                idx_previous:idx,
                simulated_params_augmented.columns.get_loc(
                    "augmentation_noise_level"
                ),
            ] = final_param_sersic[:, 3]

            simulated_params_augmented.iloc[
                idx_previous:idx,
                simulated_params_augmented.columns.get_loc("augmentation_mue"),
            ] = (
                np.ones_like(final_param_sersic[:, 0]) * mu0
            )

            simulated_params_augmented.iloc[
                idx_previous:idx,
                simulated_params_augmented.columns.get_loc("augmentation_re"),
            ] = final_param_sersic[:, 0]

            simulated_params_augmented.iloc[
                idx_previous:idx,
                simulated_params_augmented.columns.get_loc("augmentation_n"),
            ] = final_param_sersic[:, 1]

            simulated_params_augmented.iloc[
                idx_previous:idx,
                simulated_params_augmented.columns.get_loc(
                    "augmentation_r_sersic"
                ),
            ] = final_param_sersic[:, 2]

            for id in range(idx_previous, idx):
                aug = simulated_params_augmented.loc[
                    id,
                    [
                        "augmentation_mue",
                        "augmentation_re",
                        "augmentation_n",
                        "augmentation_r_sersic",
                    ],
                ].values

                # sersic_component = sersic_sb_truncated(
                #     r, aug[0], aug[1], aug[2], 1.20*aug[3], steepness=20
                # )

                sersic_component = sersic_sb(r, aug[0], aug[1], aug[2])

                noise = np.random.normal(
                    0,
                    simulated_params_augmented.loc[
                        id, "augmentation_noise_level"
                    ],
                    size=r.shape,
                )
                noise_decay = np.sqrt(10 ** (-0.4 * (26 - mu)))

                noise_clipped = np.clip(noise * noise_decay, -1.5, 2.5)

                ## if not truncated
                model = np.zeros_like(r)
                model[r < aug[3]] = sersic_component[r < aug[3]]
                model[r >= aug[3]] = mu[r >= aug[3]]

                ## If truncated
                # model = (
                #     -2.5
                #     * np.log10(
                #         10 ** (-0.4 * mu) + 10 ** (-0.4 * sersic_component)
                #     )
                #     + noise_clipped
                # )
                # Clipped values above 32 mag/arcsec^2 to avoid unrealistic values in the outskirts

                mask = model <= 32
                model = model[mask]
                r = r[mask]
                mu = mu[mask]
                sersic_component = sersic_component[mask]
                noise_clipped = noise_clipped[mask]
                model += noise_clipped

                # Save in txt
                with open(data_dir / f"profile_simulated_{id}.txt", "w") as f:
                    f.write(
                        "# Radius (arcsec)    Surface Brightness (mag/arcsec^2)\n"
                    )
                    for radius, sb in zip(r, model):
                        f.write(f"{radius:.4f}    {sb:.4f}\n")
                if True:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(r, mu, label="Exp", color="black", ls=":")

                    ax.plot(
                        r,
                        sersic_component,
                        label="Sersic",
                        color="red",
                        ls="--",
                    )
                    ax.plot(r, model, label="Combined", color="blue")
                    if label != 1:
                        for rb in rbreaks:
                            ax.plot(
                                rb,
                                mu[np.argmin(np.abs(r - rb))],
                                "x",
                                color="green",
                            )
                    ax.invert_yaxis()
                    ax.set_ylim([32, 15])
                    ax.set_xlabel("Radius (arcsec)")
                    ax.set_ylabel("Surface Brightness (mag/arcsec$^2$)")
                    ax.set_title(f"Break Type: {label}")
                    fig.savefig(
                        data_dir / f"profile_simulated_{id}.png", dpi=300
                    )
                    plt.close(fig)
            
        except Exception as e:
            failed += 1
            print(f"Error processing profile {i}: {e}")
            continue


# %%

simulated_params_augmented.to_csv(
    data_dir / "simulated_profiles_augmented_parameters.csv"
)

print(
    f"Simulation completed with {failed} failed profiles. Augmented parameters saved to CSV."
)
# %%
if False:
    # Create MP4 from the collected figures
    print(f"Creating MP4 with {len(images)} frames...")
    mp4_path = data_dir / "surface_brightness_profiles.mp4"

    # Set up the figure and animation
    fig_base = images[0]
    Writer = FFMpegWriter(
        fps=2, metadata=dict(artist="DeepDisc"), bitrate=1800
    )

    def update_frame(frame_num):
        fig_base.clear()
        # Copy the content from the stored figure
        ax = fig_base.add_subplot(111)
        source_ax = images[frame_num].axes[0]

        # Copy lines and properties
        for line in source_ax.get_lines():
            ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                label=line.get_label(),
                color=line.get_color(),
                linewidth=line.get_linewidth(),
            )

        ax.set_xlabel(source_ax.get_xlabel())
        ax.set_ylabel(source_ax.get_ylabel())
        ax.set_title(source_ax.get_title())
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax.get_children()

    anim = animation.FuncAnimation(
        fig_base,
        update_frame,
        frames=len(images),
        interval=500,  # milliseconds between frames
        repeat=True,
    )

    # Save as MP4
    anim.save(mp4_path, writer=Writer)
    print(f"MP4 saved to {mp4_path}")

    # Close all figures to free memory
    for fig in images:
        plt.close(fig)
