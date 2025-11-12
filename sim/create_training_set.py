##  This script helps create a training set for machine learning models by
##  simulating surface brightness profiles with various parameters.  ###


import os
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaincinv


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

    # Combine the components by adding them in linear normalized_sigmoid(r,x0=2*rbulge)
    total_linear = 10 ** (-0.4 * bulge) + 10 ** (-0.4 * disc_comp)
    mu = -2.5 * np.log10(total_linear)

    return mu


def normalized_sigmoid(x, x0=0):
    """
    Normalized sigmoid function.

    Parameters:
    -----------
    x : float or array-like
        Input value(s).

    Returns:
    --------
    float or array-like
        Normalized sigmoid output between 0 and 1.
    """
    return 1 - 1 / (1 + np.exp(-(x - x0)))


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


if __name__ == "__main__":
    import os
    from pathlib import Path

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Set up paths
    data_dir = Path("sim/data")
    obs_file = Path("sim/observations/breaks_2025-04-15.csv")

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

    profiles_data = []
    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(n_profiles):
        # Sample a random object from the valid breaks
        sample_idx = np.random.randint(0, len(valid_breaks))
        sample_obj = valid_breaks.iloc[sample_idx]

        # Extract parameters with some variation
        base_mue = sample_obj.get("final_mue", 22.0)
        base_re = sample_obj.get("final_re", 3.0)
        base_n = sample_obj.get("final_n", 2.0)
        base_mu0 = sample_obj.get("final_mu0", 20.0)

        # Get break parameters
        h1 = sample_obj.get("final_h1", 2.0)
        rbreak1 = sample_obj.get("final_rbreaks1", 10.0)

        # Check for second break
        h2 = sample_obj.get("final_h2", h1 * 1.5)
        rbreak2 = sample_obj.get("final_rbreaks2", None)

        # Add some random variation to parameters (±10%)
        mue = base_mue + np.random.normal(0, 0.3)
        re = max(0.5, base_re + np.random.normal(0, base_re * 0.1))
        n = max(0.3, min(8.0, base_n + np.random.normal(0, 0.2)))
        mu0 = base_mu0 + np.random.normal(0, 0.2)

        # Scale lengths and breaks
        h1_sim = max(0.5, h1 + np.random.normal(0, h1 * 0.1))
        rbreak1_sim = max(2.0, rbreak1 + np.random.normal(0, rbreak1 * 0.1))

        # Create model based on number of breaks
        if pd.notna(rbreak2) and rbreak2 > 0:
            # Two break model
            h2_sim = max(0.5, h2 + np.random.normal(0, h2 * 0.1))
            rbreak2_sim = max(
                rbreak1_sim + 2, rbreak2 + np.random.normal(0, rbreak2 * 0.1)
            )
            h_array = [h1_sim, h2_sim, h2_sim * 1.5]
            rbreaks_array = [rbreak1_sim, rbreak2_sim]
        else:
            # Single break model
            h2_sim = max(0.5, h1_sim * (1.5 + np.random.normal(0, 0.3)))
            h_array = [h1_sim, h2_sim]
            rbreaks_array = [rbreak1_sim]

        # Create the model
        try:
            model = MultiComponentModel(
                mue=mue, re=re, n=n, mu0=mu0, h=h_array, rbreaks=rbreaks_array
            )

            # Generate clean surface brightness profile
            mu_clean = model(r)

            # Add realistic noise
            # Noise increases with radius and fainter surface brightness
            noise_level = 0.1 + 0.02 * r + 0.05 * np.exp(0.1 * (mu_clean - 20))
            noise = np.random.normal(0, noise_level)
            mu_noisy = mu_clean + noise

            # Store profile data
            profile_data = {
                "profile_id": f"sim_{i+1:03d}",
                "original_object_id": sample_obj["object_id"],
                "r": r.copy(),
                "mu_clean": mu_clean.copy(),
                "mu_noisy": mu_noisy.copy(),
                "noise": noise.copy(),
                "parameters": model.get_dictionary(),
            }
            profiles_data.append(profile_data)

            # Save individual profile data
            profile_file = data_dir / f"profile_{i+1:03d}.npz"
            np.savez(
                profile_file,
                r=r,
                mu_clean=mu_clean,
                mu_noisy=mu_noisy,
                noise=noise,
            )

            # Save parameters to text file
            param_file = data_dir / f"profile_{i+1:03d}_params.txt"
            with open(param_file, "w") as f:
                f.write(f"Simulated Surface Brightness Profile {i+1:03d}\n")
                f.write(f"Generated from object: {sample_obj['object_id']}\n")
                f.write("=" * 50 + "\n\n")
                f.write("Model Parameters:\n")
                f.write("-" * 20 + "\n")
                params = model.get_dictionary()
                for key, value in params.items():
                    f.write(f"{key}: {value:.4f}\n")
                f.write(f"\nNumber of breaks: {len(rbreaks_array)}\n")
                f.write(f"Radial range: {r[0]:.2f} - {r[-1]:.2f} arcsec\n")
                f.write(f"Noise RMS: {np.std(noise):.4f} mag/arcsec²\n")

            print(f"Profile {i+1:03d} generated and saved")

        except Exception as e:
            print(f"Error generating profile {i+1}: {e}")
            continue

    print(f"\nGenerated {len(profiles_data)} profiles successfully")

    # Create animation showing all profiles
    print("Creating animation...")

    def animate(frame):
        ax.clear()

        if frame < len(profiles_data):
            profile = profiles_data[frame]
            r_data = profile["r"]
            mu_clean = profile["mu_clean"]
            mu_noisy = profile["mu_noisy"]
            params = profile["parameters"]

            # Plot the surface brightness profile
            ax.plot(
                r_data,
                mu_clean,
                "b-",
                linewidth=2,
                label="Clean model",
                alpha=0.8,
            )
            ax.plot(
                r_data,
                mu_noisy,
                "r-",
                linewidth=1,
                label="With noise",
                alpha=0.7,
            )

            # Plot individual components if possible
            try:
                model = MultiComponentModel(
                    mue=params["mue"],
                    re=params["re"],
                    n=params["n"],
                    mu0=params["mu0"],
                    h=[
                        params[k]
                        for k in params.keys()
                        if k.startswith("h") and k[1:].isdigit()
                    ],
                    rbreaks=[
                        params[k]
                        for k in params.keys()
                        if k.startswith("rbreak") and k[6:].isdigit()
                    ],
                )

                # Plot components
                disc_components = model.get_components(r_data)
                for j, disc in enumerate(disc_components):
                    ax.plot(r_data, disc, "--", alpha=0.5, label=f"Disc {j+1}")

                # Plot bulge
                bulge = sersic_sb(r_data, model.mue, model.re, model.n)
                ax.plot(r_data, bulge, ":", alpha=0.5, label="Bulge")

            except:
                pass

            ax.invert_yaxis()
            ax.set_xlabel("Radius (arcsec)")
            ax.set_ylabel("Surface Brightness (mag/arcsec²)")
            ax.set_title(
                f'Profile {frame+1:03d} - Object: {profile["original_object_id"]}\n'
                f'Breaks: {params.get("ndiscs", 1)-1} | '
                f'μₑ={params.get("mue", 0):.1f}, rₑ={params.get("re", 0):.1f}, '
                f'n={params.get("n", 0):.1f}'
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, r_max)

            # Set y-limits based on data
            valid_mu = mu_noisy[np.isfinite(mu_noisy)]
            if len(valid_mu) > 0:
                y_min = min(np.min(valid_mu) - 1, 18)
                y_max = max(np.max(valid_mu) + 1, 28)
                ax.set_ylim(y_max, y_min)

    # Create and save animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(profiles_data),
        interval=2000,
        repeat=True,
        blit=False,
    )

    # Save as MP4 video
    video_file = data_dir / "surface_brightness_profiles.mp4"
    try:
        ani.save(str(video_file), writer="ffmpeg", fps=0.5, bitrate=1800)
        print(f"Video saved as: {video_file}")
    except Exception as e:
        print(f"Could not save video (ffmpeg might not be installed): {e}")
        # Save as GIF as fallback
        gif_file = data_dir / "surface_brightness_profiles.gif"
        try:
            ani.save(str(gif_file), writer="pillow", fps=0.5)
            print(f"Animation saved as GIF: {gif_file}")
        except Exception as e2:
            print(f"Could not save GIF either: {e2}")

    # Create summary plot with all profiles
    fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles_data)))

    for i, profile in enumerate(profiles_data):
        ax_summary.plot(
            profile["r"],
            profile["mu_noisy"],
            color=colors[i],
            alpha=0.7,
            linewidth=1,
            label=f"Profile {i+1}",
        )

    ax_summary.invert_yaxis()
    ax_summary.set_xlabel("Radius (arcsec)")
    ax_summary.set_ylabel("Surface Brightness (mag/arcsec²)")
    ax_summary.set_title(
        f"All {len(profiles_data)} Simulated Surface Brightness Profiles"
    )
    ax_summary.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax_summary.grid(True, alpha=0.3)

    summary_file = data_dir / "all_profiles_summary.png"
    plt.tight_layout()
    plt.savefig(summary_file, dpi=150, bbox_inches="tight")
    print(f"Summary plot saved as: {summary_file}")

    # Save combined dataset
    combined_file = data_dir / "all_profiles_data.npz"
    all_r = np.array([p["r"] for p in profiles_data])
    all_mu_clean = np.array([p["mu_clean"] for p in profiles_data])
    all_mu_noisy = np.array([p["mu_noisy"] for p in profiles_data])
    all_noise = np.array([p["noise"] for p in profiles_data])

    np.savez(
        combined_file,
        r=all_r,
        mu_clean=all_mu_clean,
        mu_noisy=all_mu_noisy,
        noise=all_noise,
        profile_ids=[p["profile_id"] for p in profiles_data],
        object_ids=[p["original_object_id"] for p in profiles_data],
    )

    print(f"Combined dataset saved as: {combined_file}")
    print(f"\nAll files saved in: {data_dir}")

    plt.show()
