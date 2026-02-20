# Installation

## Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy >= 1.22
- SciPy >= 1.8
- Matplotlib >= 3.5

## Install from Source

Clone the repository and install:

```bash
git clone https://github.com/GeoBrain-Project/geobrain.git
cd geobrain
pip install -e ".[all]"
```

## GPU Support

GeoBrain leverages PyTorch for GPU acceleration. Install PyTorch with CUDA
support following the [official instructions](https://pytorch.org/get-started/locally/):

```bash
# Example for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU availability:

```python
import torch
print(torch.cuda.is_available())   # True if CUDA is configured
print(torch.cuda.device_count())   # Number of GPUs
```

## Verify Installation

```python
import geobrain
print(geobrain.__version__)  # 0.1.0
```

Run a quick test:

```python
from geobrain.geomodel import Simulator, SimulationConfig

config = SimulationConfig(
    shape=(64, 64),
    lh=10, lv=5,
    seed=42,
)
sim = Simulator.create("fft_ma")
field = sim.simulate(config)
print(f"Generated field shape: {field.shape}")
```
