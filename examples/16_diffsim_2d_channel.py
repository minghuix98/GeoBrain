#!/usr/bin/env python
"""
DiffSim Case 1 example for 2D channel facies: load local compact test data,
run unconditional and conditional generation, and save figures.

Run from the GeoBrain repository root or from the examples directory:
    python examples/16_diffsim_2d_channel.py
"""

# %% [markdown]
# # Case 1: 2D Channel Facies Geomodeling
#
# This notebook demonstrates both unconditional and conditional diffusion for 2D channel facies modeling.

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

import torch
import numpy as np
import matplotlib.pyplot as plt

# GeoBrain imports
from geobrain.geomodel.geogen import DiffSimSimulator
from geobrain.geomodel.geogen.diffsim.data.dataset import InpaintDatasetCase1

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

# %%
# Config path
config_path = repo_root / "configs/diffsim/case1_geomodeling.json"
checkpoint_dir = repo_root / "checkpoints/diffsim"
result_dir = repo_root / "examples/results/diffsim/case1"
result_dir.mkdir(parents=True, exist_ok=True)
print(f"Using config: {config_path}")

# %% [markdown]
# ## Section 1: Unconditional Generation

# %%
# Create unconditional simulator from config
sim_uncond = DiffSimSimulator.from_config(
    config_path=config_path,
    mode='unconditional',
    sampler='ddpm',
    device=device
)
print(f"Created unconditional simulator")
print(f"  - Model type: {sim_uncond.model_type}")
print(f"  - Image size: {sim_uncond.image_size}")
print(f"  - Timesteps: {sim_uncond.timesteps}")
print(f"  - Beta schedule: {sim_uncond.beta_schedule}")

# %%
# Generate samples using DDPM
set_seed(42)
samples_ddpm = sim_uncond.generate_unconditional(
    n_samples=16,
    device=device,
    sampler='ddpm',
    ddpm_steps=UNCOND_DDPM_STEPS
)
print(f"Generated {samples_ddpm.shape[0]} samples with shape {samples_ddpm.shape[1:]}")

# %%
# Visualize DDPM generated samples
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i < samples_ddpm.shape[0]:
        img = samples_ddpm[i, 0].cpu().numpy()
        ax.imshow(img, cmap='viridis')
    ax.axis('off')
plt.suptitle(f'DDPM Generated Samples ({UNCOND_DDPM_STEPS} steps)', fontsize=16)
plt.tight_layout()
plt.savefig(result_dir / 'FigureCase1_Uncond_DDPM.png', dpi=300)
plt.show()

# %% [markdown]
# ### DDIM Sampling (Faster)

# %%
# Generate samples using DDIM (faster than full DDPM)
set_seed(40)
samples_ddim = sim_uncond.generate_unconditional(
    n_samples=16,
    device=device,
    sampler='ddim',
    ddim_steps=UNCOND_DDIM_STEPS,
    eta=0.0  # deterministic
)
print(f"Generated {samples_ddim.shape[0]} DDIM samples")

# %%
# Visualize DDIM generated samples
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i < samples_ddim.shape[0]:
        img = samples_ddim[i, 0].cpu().numpy()
        ax.imshow(img, cmap='viridis')
    ax.axis('off')
plt.suptitle(f'DDIM Generated Samples ({UNCOND_DDIM_STEPS} steps)', fontsize=16)
plt.tight_layout()
plt.savefig(result_dir / 'FigureCase1_Uncond_DDIM.png', dpi=300)
plt.show()

# %% [markdown]
# ## Section 2: Conditional Generation (Inpainting with Sparse Well Conditions)
#
# The conditional model takes sparse well observations as conditioning input.

# %%
# Create conditional simulator from config
sim_cond = DiffSimSimulator.from_config(
    config_path=config_path,
    mode='conditional',
    sampler='ddpm',
    device=device
)
sim_cond.beta_schedule_config.setdefault('test', {})['n_timestep'] = COND_DDPM_STEPS
print(f"Created conditional simulator")
print(f"  - Model type: {sim_cond.model_type}")
print(f"  - Image size: {sim_cond.image_size}")
print(f"  - Test schedule timesteps: {sim_cond.beta_schedule_config['test']['n_timestep']}")

# %% [markdown]
# ### Load Dataset with InpaintDataset

# %%
# Load config for data paths
import json
with open(config_path) as f:
    config = json.load(f)

# Define data paths from config
cond_data = config.get('conditional', {}).get('data', {})
images_path = resolve_repo_path(cond_data.get('test_image', ''))
masks_path = resolve_repo_path(cond_data.get('test_mask', ''))
data_root = (images_path, masks_path)

# Mask configuration
mask_config = {
    'mask_mode': 'file'  # Use mask files from the masks directory
}

# Load dataset
dataset = InpaintDatasetCase1(data_root, mask_config=mask_config, data_len=-1, image_size=[64, 64])
print(f"Dataset size: {len(dataset)}")

# %% [markdown]
# ### Visualize One Sample

# %%
# Plot one sample from the dataset
plt.figure(figsize=(15, 3))
inum = 4

plt.subplot(151)
plt.imshow(dataset[inum]['gt_image'].cpu().numpy().reshape(64, 64))
plt.colorbar()
plt.title('Test image')

plt.subplot(152)
plt.imshow(dataset[inum]['yt_image'].cpu().numpy().reshape(64, 64))
plt.colorbar()
plt.title('T images')

plt.subplot(153)
plt.imshow(dataset[inum]['mask_image'].cpu().numpy().reshape(64, 64))
plt.colorbar()
plt.title('Mask images')

plt.subplot(154)
plt.imshow(dataset[inum]['mask'].cpu().numpy().reshape(64, 64), 'gray')
plt.colorbar()
plt.title('Mask')

plt.subplot(155)
plt.imshow(dataset[inum]['cond_image'][3])
plt.colorbar()
plt.title('Conditional images')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Generate Multiple Realizations for Test Samples

# %%
# Local test sample indices for generation
sample_indices = list(range(8))
numlist = sample_indices
num_realizations = 10

print(f"Generating {num_realizations} realizations for {len(numlist)} test samples...")

# %%
# Generate realizations for each test sample using DiffSimSimulator
set_seed(42)

items = [dataset[inum] for inum in numlist]
gt_batch = torch.stack([item['gt_image'] for item in items], dim=0)
mask_batch = torch.stack([item['mask'] for item in items], dim=0)
cond_batch = torch.stack([item['cond_image'] for item in items], dim=0)

cond_input = cond_batch.repeat_interleave(num_realizations, dim=0)
gt_image_batch = gt_batch.repeat_interleave(num_realizations, dim=0)
mask_input = mask_batch.repeat_interleave(num_realizations, dim=0)
yt_input = gt_image_batch * (1. - mask_input) + mask_input * torch.randn_like(gt_image_batch)

print(f"Running DDPM conditional batch with {cond_input.shape[0]} realizations and {sim_cond.beta_schedule_config['test']['n_timestep']} steps...")
output, visuals = sim_cond.generate_conditional(
    y_cond=cond_input,
    y_t=yt_input,
    y_0=gt_image_batch,
    mask=mask_input,
    device=device,
    sampler='ddpm',
    ddpm_steps=COND_DDPM_STEPS,
    sample_num=10
)

total_output = output.cpu().numpy().reshape(len(numlist), num_realizations, 64, 64)

print("Generation complete!")
print(f"Output shape: {total_output.shape}")

# %%
# Compute mean probability for each facies across realizations
# Facies values (normalized to [-1, 1]): Sand ~1, Bank ~0, Mud ~-1

# Define thresholds for facies classification
sand_threshold = 0.5    # Values > 0.5 are sand
mud_threshold = -0.5    # Values < -0.5 are mud
# Bank is in between

# Compute binary masks for each facies across all realizations
sand_masks = (total_output > sand_threshold).astype(np.float32)  # [num_samples, num_realizations, H, W]
mud_masks = (total_output < mud_threshold).astype(np.float32)
bank_masks = ((total_output >= mud_threshold) & (total_output <= sand_threshold)).astype(np.float32)

# Compute mean probability (frequency) for each facies
mean_sand = np.mean(sand_masks, axis=1)  # [num_samples, H, W]
mean_bank = np.mean(bank_masks, axis=1)
mean_mud = np.mean(mud_masks, axis=1)

print(f"Mean sand shape: {mean_sand.shape}")
print(f"Mean bank shape: {mean_bank.shape}")
print(f"Mean mud shape: {mean_mud.shape}")

# %% [markdown]
# ### Plot Results: Test Images, Realizations, Mean, and Variance

# %%
# Create output directory
dir_name = str(result_dir)

import matplotlib.cm as cm
import matplotlib.patches as mpatches

fig, ax = plt.subplots(8, 11, sharex='col', sharey='row')
fig.set_size_inches(11, 8, forward=True)

for i in range(8):
    inum = numlist[i]
    re_mask = 1 - dataset[inum]['mask'].cpu().numpy().reshape(64, 64).astype(np.float32)
    re_mask[re_mask == 0] = np.nan
    gt = dataset[inum]['gt_image'].cpu().numpy().reshape(64, 64)
    gt[np.isnan(re_mask)] = np.nan

    ax[i, 0].imshow(dataset[inum]['gt_image'].cpu().numpy().reshape(64, 64))
    ax[i, 1].imshow(gt)
    # Scatter plot on top of the facies image to enlarge points
    facies = gt.copy()
    y, x = np.where(~np.isnan(facies))
    ax[i, 1].scatter(
        x, y, c=facies[~np.isnan(facies)], cmap='viridis',
        s=32, marker='s', vmin=-1, vmax=1, edgecolors='black', linewidths=0.2
    )

    for j in range(6):
        ax[i, j+2].imshow(total_output[i, j, :, :])

    # Column 8: Channel Sand (mean probability)
    ax[i, 8].imshow(mean_sand[i], cmap='jet', vmin=0, vmax=1)

    # Column 9: Channel Bank (mean probability)
    ax[i, 9].imshow(mean_bank[i], cmap='jet', vmin=0, vmax=1)

    # Column 10: Inter-channel Mud (mean probability)
    h = ax[i, 10].imshow(mean_mud[i], cmap='jet', vmin=0, vmax=1)

ax[0, 0].set_title('Test faceis\nmodel', fontsize=10)
ax[0, 1].set_title('Conditioning\nfacies', fontsize=10)
ax[0, 5].set_title('Realizations')
ax[0, -1].set_title('Inter-channel\nMud', fontsize=10)
ax[0, -2].set_title('Channel\nBank', fontsize=10)
ax[0, -3].set_title('Channel\nSand', fontsize=10)

# Hide labels but keep ticks
for i in range(8):
    for j in range(11):
        ax[i, j].tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
right = 0.87
plt.subplots_adjust(left=0.02, bottom=0.08, right=right, top=0.94, wspace=0.12, hspace=0.12)

# Create custom axes for the colorbars on the right of the plot
cbaxes_mean = fig.add_axes([right + 0.02, 0.18, 0.018, 0.62])

# Add colorbars
cbar_mean = fig.colorbar(h, orientation='vertical', cax=cbaxes_mean)

# Set labels for the colorbars
cbar_mean.set_label('Mean of facies')

# Sample colors from the 'viridis' colormap
cmap = cm.get_cmap('viridis')
channel_mud_color = cmap(0.0)   # -1: Channel Mud
channel_bank_color = cmap(0.5)  # 0: Channel Bank
channel_sand_color = cmap(1.0)  # 1: Channel Sand

# Create custom legend handles with square patches
channel_mud = mpatches.Patch(color=channel_mud_color, label='Inter-channel Mud')
channel_bank = mpatches.Patch(color=channel_bank_color, label='Channel Bank')
channel_sand = mpatches.Patch(color=channel_sand_color, label='Channel Sand')

# Add the legend to the figure using fig.legend
legend = fig.legend(handles=[channel_mud, channel_bank, channel_sand], loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

plt.savefig(
    dir_name + '/FigureCase1_Cond_DDPM.png',
    dpi=300,
    bbox_extra_artists=(legend,),
    bbox_inches='tight',
    pad_inches=0.1
)
plt.show()

print(f"Figure saved to {dir_name}/FigureCase1_Cond_DDPM.png")

# %% [markdown]
# ### DDIM Sampling for Conditional Generation (Faster)

# %%
# Generate realizations using DDIM over the full test schedule with selected steps
set_seed(40)

print(f"Generating {num_realizations} realizations for {len(numlist)} test samples using DDIM...")

yt_input = gt_image_batch * (1. - mask_input) + mask_input * torch.randn_like(gt_image_batch)
output, visuals = sim_cond.generate_conditional(
    y_cond=cond_input,
    y_t=yt_input,
    y_0=gt_image_batch,
    mask=mask_input,
    device=device,
    sampler='ddim',
    ddim_steps=COND_DDIM_STEPS,
    eta=0.0,
    sample_num=8
)

total_output_ddim = output.cpu().numpy().reshape(len(numlist), num_realizations, 64, 64)

print("DDIM Generation complete!")
print(f"Output shape: {total_output_ddim.shape}")

# %%
# Compute mean probability for each facies across DDIM realizations
sand_masks_ddim = (total_output_ddim > sand_threshold).astype(np.float32)
mud_masks_ddim = (total_output_ddim < mud_threshold).astype(np.float32)
bank_masks_ddim = ((total_output_ddim >= mud_threshold) & (total_output_ddim <= sand_threshold)).astype(np.float32)

mean_sand_ddim = np.mean(sand_masks_ddim, axis=1)
mean_bank_ddim = np.mean(bank_masks_ddim, axis=1)
mean_mud_ddim = np.mean(mud_masks_ddim, axis=1)

print(f"DDIM Mean sand shape: {mean_sand_ddim.shape}")

# %% [markdown]
# ### Plot DDIM Results

# %%
# Plot DDIM results: Test Images, Realizations, Mean Facies Probabilities
fig, ax = plt.subplots(8, 11, sharex='col', sharey='row')
fig.set_size_inches(11, 8, forward=True)

for i in range(8):
    inum = numlist[i]
    re_mask = 1 - dataset[inum]['mask'].cpu().numpy().reshape(64, 64).astype(np.float32)
    re_mask[re_mask == 0] = np.nan
    gt = dataset[inum]['gt_image'].cpu().numpy().reshape(64, 64)
    gt_cond = gt.copy()
    gt_cond[np.isnan(re_mask)] = np.nan

    ax[i, 0].imshow(gt)
    ax[i, 1].imshow(gt_cond)
    y, x = np.where(~np.isnan(gt_cond))
    ax[i, 1].scatter(
        x, y, c=gt_cond[~np.isnan(gt_cond)], cmap='viridis',
        s=32, marker='s', vmin=-1, vmax=1, edgecolors='black', linewidths=0.2
    )

    for j in range(6):
        ax[i, j+2].imshow(total_output_ddim[i, j, :, :])

    ax[i, 8].imshow(mean_sand_ddim[i], cmap='jet', vmin=0, vmax=1)
    ax[i, 9].imshow(mean_bank_ddim[i], cmap='jet', vmin=0, vmax=1)
    h = ax[i, 10].imshow(mean_mud_ddim[i], cmap='jet', vmin=0, vmax=1)

ax[0, 0].set_title('Test facies\nmodel', fontsize=10)
ax[0, 1].set_title('Conditioning\nfacies', fontsize=10)
ax[0, 5].set_title('DDIM Realizations')
ax[0, -1].set_title('Inter-channel\nMud', fontsize=10)
ax[0, -2].set_title('Channel\nBank', fontsize=10)
ax[0, -3].set_title('Channel\nSand', fontsize=10)

for i in range(8):
    for j in range(11):
        ax[i, j].tick_params(labelbottom=False, labelleft=False)

plt.tight_layout()
right = 0.87
plt.subplots_adjust(left=0.02, bottom=0.08, right=right, top=0.94, wspace=0.12, hspace=0.12)

cbaxes_mean = fig.add_axes([right + 0.02, 0.18, 0.018, 0.62])
cbar_mean = fig.colorbar(h, orientation='vertical', cax=cbaxes_mean)
cbar_mean.set_label('Mean of facies')

legend = fig.legend(handles=[channel_mud, channel_bank, channel_sand], loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

plt.savefig(
    dir_name + '/FigureCase1_Cond_DDIM.png',
    dpi=300,
    bbox_extra_artists=(legend,),
    bbox_inches='tight',
    pad_inches=0.1
)
plt.show()

print(f"DDIM Figure saved to {dir_name}/FigureCase1_Cond_DDIM.png")
