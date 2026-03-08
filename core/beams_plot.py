"""Complex field visualization: 2x2 subplot (Real, Imag, Amplitude, Phase)."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List


def show_complex_plot(
    cplx_arr: np.ndarray,
    meta: Dict[str, Any],
    title_prefix: str = "",
    pos: List[List[int]] = None,
) -> plt.Figure:
    """
    Plot complex field as 2x2: Real, Imaginary, Amplitude, Phase.
    Returns matplotlib Figure for use with st.pyplot(fig).
    """
    if pos is None:
        pos = [[0, 0], [0, 1], [1, 0], [1, 1]]

    extent = None
    xlabel, ylabel = "x (px)", "y (px)"

    dx = meta.get("dx")
    dy = meta.get("dy")
    nx = meta.get("nx")
    ny = meta.get("ny")
    if dx is not None and dy is not None and nx is not None and ny is not None and nx > 0 and ny > 0:
        width = nx * dx
        height = ny * dy
        extent = [0, width, 0, height]
        xlabel, ylabel = "x (µm)", "y (µm)"

    real_part = np.real(cplx_arr)
    imag_part = np.imag(cplx_arr)
    amplitude = np.abs(cplx_arr)
    phase = np.angle(cplx_arr)

    max_x = max(p[0] for p in pos)
    max_y = max(p[1] for p in pos)
    fig, axes = plt.subplots(max_x + 1, max_y + 1, figsize=(12, 10))
    fig.suptitle(f"Field Analysis: {title_prefix}", fontsize=16)

    is_1d_layout = max_x == 0 or max_y == 0
    if max_x == 0:
        pos_flat = [p[1] for p in pos]
    elif max_y == 0:
        pos_flat = [p[0] for p in pos]
    else:
        pos_flat = None

    def get_ax(i: int):
        if is_1d_layout and pos_flat is not None:
            return axes[pos_flat[i]]
        return axes[pos[i][0], pos[i][1]]

    ax = get_ax(0)
    im = ax.imshow(real_part, origin="lower", extent=extent, cmap="RdBu_r")
    ax.set_title("Real Part")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)

    ax = get_ax(1)
    im = ax.imshow(imag_part, origin="lower", extent=extent, cmap="RdBu_r")
    ax.set_title("Imaginary Part")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)

    ax = get_ax(2)
    im = ax.imshow(amplitude, origin="lower", extent=extent, cmap="inferno")
    ax.set_title("Amplitude")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)

    ax = get_ax(3)
    im = ax.imshow(phase, origin="lower", extent=extent, cmap="twilight")
    ax.set_title("Phase (rad)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig
