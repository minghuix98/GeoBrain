#!/usr/bin/env python
"""
DiffSim Case 3 example for 3D LA facies: load local compact test volumes,
run unconditional and conditional generation, and save slice figures.

Run from the GeoBrain repository root or from the examples directory:
    python examples/18_diffsim_3d_facies.py
"""

# %% [markdown]
# # Case 3: 3D Facies Modeling (LA3D)
#
# This notebook demonstrates both unconditional and conditional diffusion for 3D facies modeling.
#
# Volume size: [32, 48, 48] (D, H, W)
#
# For visualization, we show 2D sections (slices) rather than 3D cubes.

# %%
import sys
import os
from pathlib import Path

def find_repo_root(start=None):
    start = Path.cwd() if start is None else Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "geobrain").is_dir():
            return path
    raise RuntimeError("Run this notebook from inside the GeoBrain repository.")

repo_root = find_repo_root(Path(__file__).resolve().parent) if "__file__" in globals() else find_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def resolve_repo_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else repo_root / path

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm, ListedColormap

# GeoBrain imports
from geobrain.geomodel.geogen import DiffSimSimulator
from geobrain.geomodel.geogen.diffsim.data.dataset import NPYInpaintDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


UNCOND_DDPM_STEPS = int(os.environ.get("DIFFSIM_UNCOND_DDPM_STEPS", "1500"))
COND_DDPM_STEPS = int(os.environ.get("DIFFSIM_COND_DDPM_STEPS", os.environ.get("DIFFSIM_DDPM_STEPS", "1500")))
UNCOND_DDIM_STEPS = 60
COND_DDIM_STEPS = 60
NUM_REALIZATIONS = 5

FACIES_CMAP = ListedColormap(plt.get_cmap('viridis')(np.linspace(0.0, 1.0, 4)))
FACIES_CMAP.set_bad(color='white')
FACIES_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], FACIES_CMAP.N)
FACIES_LABELS = ['Floodplain', 'Mud drapes', 'Channel fill', 'Lateral accretion sand']
COND_FACIES_THRESHOLDS = [-2.0 / 3.0, 0.0, 2.0 / 3.0]
UNCOND_FACIES_THRESHOLDS = [1.0 / 6.0, 0.5, 5.0 / 6.0]


def discretize_facies(values, scale='conditional'):
    """Map continuous facies values to four discrete classes for plotting."""
    values = np.asarray(values)
    if scale == 'unconditional':
        values = np.clip(values, 0.0, 1.0)
        thresholds = UNCOND_FACIES_THRESHOLDS
    elif scale == 'conditional':
        values = np.clip(values, -1.0, 1.0)
        thresholds = COND_FACIES_THRESHOLDS
    else:
        raise ValueError("scale must be 'conditional' or 'unconditional'")
    return np.digitize(values, thresholds).astype(np.int8)


def plot_discrete(ax, values, mask=None, aspect=None, scale='conditional'):
    classes = discretize_facies(values, scale=scale)
    if mask is not None:
        classes = np.ma.masked_where(mask > 0.5, classes)
    return ax.imshow(
        classes,
        cmap=FACIES_CMAP,
        norm=FACIES_NORM,
        aspect=aspect,
        interpolation='nearest',
    )


def plot_conditioning(ax, values, mask, aspect=None):
    plot_discrete(ax, values, mask=mask, aspect=aspect, scale='conditional')
    classes = discretize_facies(values, scale='conditional')
    known = mask < 0.5
    y, x = np.where(known)
    if len(x) > 0:
        ax.scatter(
            x, y, c=classes[known], cmap=FACIES_CMAP, norm=FACIES_NORM,
            s=8, marker='s', edgecolors='black', linewidths=0.12
        )


def facies_probabilities(realizations):
    classes = discretize_facies(realizations, scale='conditional')
    return np.stack([(classes == i).mean(axis=0) for i in range(4)], axis=0)


def add_facies_legend(fig):
    handles = [
        mpatches.Patch(color=FACIES_CMAP(FACIES_NORM(i)), label=label)
        for i, label in enumerate(FACIES_LABELS)
    ]
    fig.legend(
        handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.01),
        ncol=4, frameon=False, fontsize=8
    )


def remove_stale_case3_sample_figures(result_dir):
    for stale_file in result_dir.glob('FigureCase3_Cond_*_sample_*_section.png'):
        stale_file.unlink()


def remove_stale_uncond_section_figure(result_dir):
    stale_file = result_dir / 'FigureCase3_Uncond_DDIM_Sections.png'
    if stale_file.exists():
        stale_file.unlink()


def plot_unconditional_sections(samples, sampler_name, output_path):
    n_samples = min(4, samples.shape[0])
    sample_shape = samples.shape[2:]
    if len(sample_shape) != 3:
        raise ValueError("Expected unconditional 3D samples with shape [N, C, D, H, W]")

    d, h, w = sample_shape
    section_specs = [
        ('XY', 10, lambda volume: volume[10], None),
        ('XZ', h // 2, lambda volume: volume[:, h // 2, :], 'auto'),
        ('YZ', w // 2, lambda volume: volume[:, :, w // 2], 'auto'),
    ]
    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 2.6 * n_samples), squeeze=False)

    for row in range(n_samples):
        volume = samples[row, 0].cpu().numpy()
        for col, (section_name, section_index, selector, aspect) in enumerate(section_specs):
            ax = axes[row, col]
            plot_discrete(ax, selector(volume), aspect=aspect, scale='unconditional')
            if row == 0:
                ax.set_title(f'{section_name} section\nindex={section_index}', fontsize=10)
            ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    steps = UNCOND_DDPM_STEPS if sampler_name == 'DDPM' else UNCOND_DDIM_STEPS
    fig.suptitle(f'{sampler_name} 3D Samples ({steps} steps)', fontsize=14)
    plt.tight_layout(rect=(0.01, 0.02, 0.99, 0.95))
    plt.savefig(output_path, dpi=220)
    plt.show()
    plt.close(fig)


def plot_conditional_overview(
    outputs,
    gt_volumes,
    mask_volumes,
    sampler_name,
    section,
    output_path,
    max_realizations=5,
):
    n_rows = len(gt_volumes)
    n_realizations = min(max_realizations, outputs.shape[1])
    n_cols = 2 + n_realizations + 4
    fig, axes = plt.subplots(n_rows, n_cols, sharex='col', sharey='row', squeeze=False)
    fig.set_size_inches(1.18 * n_cols, 1.0 * n_rows, forward=True)

    prob_im = None
    for row in range(n_rows):
        gt = gt_volumes[row][0]
        mask = mask_volumes[row][0]
        row_outputs = outputs[row]

        if section == 'xy':
            slice_index = 5
            gt_section = gt[slice_index]
            mask_section = mask[slice_index]
            realization_sections = row_outputs[:, slice_index]
            prob_maps = facies_probabilities(row_outputs[:, slice_index])
            aspect = None
        elif section == 'xz':
            slice_index = int(np.argmax(np.sum(mask < 0.5, axis=(0, 2))))
            gt_section = gt[:, slice_index, :]
            mask_section = mask[:, slice_index, :]
            realization_sections = row_outputs[:, :, slice_index, :]
            prob_maps = facies_probabilities(row_outputs[:, :, slice_index, :])
            aspect = 'auto'
        else:
            raise ValueError("section must be 'xy' or 'xz'")

        plot_discrete(axes[row, 0], gt_section, aspect=aspect, scale='conditional')
        plot_conditioning(axes[row, 1], gt_section, mask_section, aspect=aspect)

        for col in range(n_realizations):
            plot_discrete(
                axes[row, 2 + col],
                realization_sections[col],
                aspect=aspect,
                scale='conditional'
            )

        for cls in range(4):
            prob_im = axes[row, 2 + n_realizations + cls].imshow(
                prob_maps[cls], cmap='jet', vmin=0, vmax=1, aspect=aspect,
                interpolation='nearest'
            )

    axes[0, 0].set_title('Test facies\nmodel', fontsize=7)
    axes[0, 1].set_title('Conditioning\nfacies', fontsize=7)
    axes[0, 2 + n_realizations // 2].set_title(f'{sampler_name}\nrealizations', fontsize=7)
    for cls, label in enumerate(FACIES_LABELS):
        axes[0, 2 + n_realizations + cls].set_title(label.replace(' ', '\n'), fontsize=7)

    for ax in axes.flat:
        ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    section_label = section.upper()
    fig.suptitle(f'{sampler_name} conditional {section_label} sections', fontsize=11)
    right = 0.95
    plt.tight_layout(rect=(0.01, 0.05, right, 0.95))
    cax = fig.add_axes([right + 0.018, 0.18, 0.016, 0.62])
    cbar = fig.colorbar(prob_im, cax=cax, orientation='vertical')
    cbar.set_label('Facies probability')
    add_facies_legend(fig)

    # plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.show()
    plt.close(fig)

# %%
# Config path
config_path = repo_root / "configs/diffsim/case3_la3d.json"
checkpoint_dir = repo_root / "checkpoints/diffsim"
result_dir = repo_root / "examples/results/diffsim/case3"
result_dir.mkdir(parents=True, exist_ok=True)
remove_stale_uncond_section_figure(result_dir)

# Load config for data paths and image size
with open(config_path) as f:
    config = json.load(f)
print(f"Using config: {config['name']}")
print(f"Volume size: {config['image_size']}")

# %% [markdown]
# ## Section 1: Unconditional Generation

# %%
# Create unconditional 3D simulator from config
sim_uncond = DiffSimSimulator.from_config(
    config_path=config_path,
    mode='unconditional',
    sampler='ddpm',
    device=device
)
print(f"Created unconditional 3D simulator")
print(f"  - Model type: {sim_uncond.model_type}")
print(f"  - Image size: {sim_uncond.image_size}")
print(f"  - Timesteps: {sim_uncond.timesteps}")
print(f"  - Beta schedule: {sim_uncond.beta_schedule}")

# %% [markdown]
# ### DDPM Sampling

# %%
# Generate 3D samples using DDPM
set_seed(42)
image_size = tuple(config['image_size'])  # (D, H, W)
samples_ddpm = sim_uncond.generate_unconditional(
    n_samples=4,
    image_size=image_size,
    device=device,
    sampler='ddpm',
    ddpm_steps=UNCOND_DDPM_STEPS
)
print(f"Generated {samples_ddpm.shape[0]} DDPM 3D samples")

# %%
# Visualize DDPM 3D samples with orthogonal sections.
plot_unconditional_sections(samples_ddpm, 'DDPM', result_dir / 'FigureCase3_Uncond_DDPM.png')

# %% [markdown]
# ### DDIM Sampling

# %%
# Generate 3D samples using DDIM
set_seed(42)
image_size = tuple(config['image_size'])  # (D, H, W)
samples_ddim = sim_uncond.generate_unconditional(
    n_samples=4,
    image_size=image_size,
    device=device,
    sampler='ddim',
    ddim_steps=UNCOND_DDIM_STEPS,
    eta=0.0  # deterministic
)
print(f"Generated {samples_ddim.shape[0]} DDIM 3D samples")

# %%
# Visualize DDIM 3D samples with orthogonal sections.
plot_unconditional_sections(samples_ddim, 'DDIM', result_dir / 'FigureCase3_Uncond_DDIM.png')

# %% [markdown]
# ## Section 2: Conditional Generation (3D Inpainting)
#
# The conditional model takes sparse observations as conditioning input.
#
# **Note:** The conditional dataset and outputs are normalized to [-1, 1], while unconditional 3D samples are visualized on the [0, 1] scale used by their training data.

# %%
# Create conditional 3D simulator from config
sim_cond = DiffSimSimulator.from_config(
    config_path=config_path,
    mode='conditional',
    sampler='ddpm',
    device=device
)
sim_cond.beta_schedule_config.setdefault('test', {})['n_timestep'] = COND_DDPM_STEPS
print(f"Created conditional 3D simulator")
print(f"  - Model type: {sim_cond.model_type}")
print(f"  - Image size: {sim_cond.image_size}")
print(f"  - Predict type: {sim_cond.predict_type}")  # x_start for 3D case
print(f"  - Test schedule timesteps: {sim_cond.beta_schedule_config['test']['n_timestep']}")

# %% [markdown]
# ### Load 3D Dataset

# %%
# Define data paths from config (support both nested and flat config structure)
cond_data = config.get('conditional', {}).get('data', config.get('data', {}))
images_path = resolve_repo_path(cond_data.get('test_image', cond_data.get('test_image_path', '')))
masks_path = resolve_repo_path(cond_data.get('test_mask', cond_data.get('test_mask_path', '')))
data_root = (images_path, masks_path)

# Mask configuration
mask_config = {
    'mask_mode': 'file'  # Use mask files from the masks directory
}

# Load dataset
dataset = NPYInpaintDataset(data_root, mask_config=mask_config, image_size=config['image_size'])
print(f"Dataset size: {len(dataset)}")

# %% [markdown]
# ### Set Up Conditional Samples

# %%
# Local test sample indices for generation
sample_indices = list(range(8))
numlist = sample_indices
num_realizations = NUM_REALIZATIONS

d, h, w = config['image_size']
dir_name = str(result_dir)
n_samples = len(numlist)
gt_volumes = []
mask_volumes = []
remove_stale_case3_sample_figures(result_dir)

for inum in numlist:
    data = dataset[inum]
    gt_volumes.append(data['gt_image'].cpu().numpy())
    mask_volumes.append(data['mask'].cpu().numpy())

print(f"Generating {num_realizations} realizations for {len(numlist)} test samples...")

# %% [markdown]
# ### Generate Multiple Realizations with DDPM

# %%
# Generate realizations using DDPM
set_seed(42)

print(f"Generating {num_realizations} realizations for {len(numlist)} test samples using DDPM...")

items = [dataset[inum] for inum in numlist]
gt_batch = torch.stack([item['gt_image'] for item in items], dim=0)
mask_batch = torch.stack([item['mask'] for item in items], dim=0)
cond_batch = torch.stack([item['cond_image'] for item in items], dim=0)

cond_input = cond_batch.repeat_interleave(num_realizations, dim=0)
gt_input = gt_batch.repeat_interleave(num_realizations, dim=0)
mask_input = mask_batch.repeat_interleave(num_realizations, dim=0)
yt_input = gt_input * (1. - mask_input) + mask_input * torch.randn_like(gt_input)

print(f"Running DDPM conditional batch with {cond_input.shape[0]} realizations and {sim_cond.beta_schedule_config['test']['n_timestep']} steps...")
output, visuals = sim_cond.generate_conditional(
    y_cond=cond_input,
    y_t=yt_input,
    y_0=gt_input,
    mask=mask_input,
    device=device,
    sampler='ddpm',
    ddpm_steps=COND_DDPM_STEPS,
    sample_num=8
)

total_output = output.cpu().numpy().reshape(len(numlist), num_realizations, d, h, w)

print("DDPM Generation complete!")
print(f"Output shape: {total_output.shape}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% [markdown]
# ### Plot DDPM Conditional Overview Figures

# %%
plot_conditional_overview(
    total_output, gt_volumes, mask_volumes, 'DDPM', 'xy',
    result_dir / 'FigureCase3_Cond_DDPM_XY.png',
    max_realizations=NUM_REALIZATIONS
)
plot_conditional_overview(
    total_output, gt_volumes, mask_volumes, 'DDPM', 'xz',
    result_dir / 'FigureCase3_Cond_DDPM_XZ.png',
    max_realizations=NUM_REALIZATIONS
)

print(f"DDPM overview figures saved to {dir_name}/")

# %% [markdown]
# ### Generate Multiple Realizations with DDIM

# %%
# Generate realizations using DDIM
set_seed(40)

print(f"Generating {num_realizations} realizations for {len(numlist)} test samples using DDIM...")

yt_input = gt_input * (1. - mask_input) + mask_input * torch.randn_like(gt_input)
output, visuals = sim_cond.generate_conditional(
    y_cond=cond_input,
    y_t=yt_input,
    y_0=gt_input,
    mask=mask_input,
    device=device,
    sampler='ddim',
    ddim_steps=COND_DDIM_STEPS,
    eta=0.0,
    sample_num=8
)

total_output_ddim = output.cpu().numpy().reshape(len(numlist), num_realizations, d, h, w)

print("DDIM Generation complete!")
print(f"Output shape: {total_output_ddim.shape}")

# %% [markdown]
# ### Plot DDIM Conditional Overview Figures

# %%
plot_conditional_overview(
    total_output_ddim, gt_volumes, mask_volumes, 'DDIM', 'xy',
    result_dir / 'FigureCase3_Cond_DDIM_XY.png',
    max_realizations=NUM_REALIZATIONS
)
plot_conditional_overview(
    total_output_ddim, gt_volumes, mask_volumes, 'DDIM', 'xz',
    result_dir / 'FigureCase3_Cond_DDIM_XZ.png',
    max_realizations=NUM_REALIZATIONS
)

print(f"DDIM overview figures saved to {dir_name}/")
