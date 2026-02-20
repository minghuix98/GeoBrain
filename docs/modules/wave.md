# Wave Physics (`geobrain.physics.wave`)

The wave module provides a complete toolkit for seismic wave propagation
simulation, AVO reflectivity modeling, wavelet generation, and seismic data
management.

## Acoustic Forward Modeling

### Grid & Boundary Configuration

```python
from geobrain.physics.wave import GridConfig, BoundaryConfig

grid = GridConfig(nx=200, nz=100, dx=10.0, dz=10.0)

# PML boundary (recommended)
boundary = BoundaryConfig(type='pml', n_layers=20, free_surface=True)

# Alternatives
boundary_gerjan = BoundaryConfig(type='gerjan', n_layers=20, alpha=0.005)
boundary_sincos = BoundaryConfig(type='sincos', n_layers=20)
```

### Velocity Model

```python
from geobrain.physics.wave import AcousticModel
import numpy as np

vp = np.ones((100, 200)) * 2500.0   # m/s
rho = np.ones((100, 200)) * 2000.0  # kg/m3

model = AcousticModel(
    grid=grid,
    boundary=boundary,
    vp=vp,
    rho=rho,
    vp_grad=True,    # Enable gradient for inversion
    device='cuda',
)
```

### Survey Setup

```python
from geobrain.physics.wave import Source, Receiver, Survey, RickerWavelet

# Generate wavelet
ricker = RickerWavelet()
wavelet, t = ricker(f0=15.0, dt=0.001)
wavelet_np = wavelet.numpy()

# Pad to simulation length
wavelet_padded = np.pad(wavelet_np, (0, NT - len(wavelet_np)))

# Source and receivers
source = Source(nt=NT, dt=DT, f0=15.0)
source.add_source(x=100, z=5, wavelet=wavelet_padded)

receiver = Receiver(nt=NT, dt=DT)
receiver.add_receivers(
    x=np.arange(0, 200),
    z=np.ones(200, dtype=int) * 5,
    receiver_type='pr',
)

survey = Survey(source, receiver)
```

### Propagation

```python
from geobrain.physics.wave import AcousticPropagator

propagator = AcousticPropagator(model, survey, device='cuda')
result = propagator.forward(checkpoint_segments=4)

pressure = result['p']              # (n_shots, nt, n_receivers)
wavefield = result['forward_wavefield_p']  # Last snapshot
```

```{figure} ../../examples/figs/acoustic_wavefield.gif
:width: 100%
:name: fig-acoustic-wavefield

Acoustic wavefield propagation through the Marmousi2 model.
```

## Elastic Modeling

```python
from geobrain.physics.wave import IsotropicElasticModel

model = IsotropicElasticModel(
    grid=grid,
    boundary=boundary,
    vp=vp_array,
    vs=vs_array,
    rho=rho_array,
)
model.forward()  # Computes Lame parameters (mu, lam, lamu, b)

poisson = model.get_poisson_ratio()
vp_vs = model.get_vp_vs_ratio()
```

## Wavelets

### Class-Based API

All wavelets return `(wavelet_tensor, time_axis_tensor)`:

```python
from geobrain.physics.wave import (
    RickerWavelet, GaussianWavelet, OrmsbyWavelet, KlauderWavelet,
    create_wavelet,
)

ricker = RickerWavelet()
wavelet, t = ricker(f0=25.0, dt=0.001)

gaussian = GaussianWavelet()
wavelet, t = gaussian(f0=25.0, dt=0.001)

ormsby = OrmsbyWavelet(f1=5, f2=10, f3=40, f4=60)
wavelet, t = ormsby(f0=0, dt=0.001)

klauder = KlauderWavelet(f_low=10, f_high=80, T=8.0)
wavelet, t = klauder(f0=0, dt=0.002)

# Factory function
wavelet, t = create_wavelet('ricker', f0=25.0, dt=0.001)
```

```{figure} ../../examples/figs/06_ricker_wavelets.png
:width: 100%
:name: fig-ricker-wavelets

Ricker wavelets at different dominant frequencies.
```

### Functional API (NumPy)

```python
from geobrain.physics.wave import (
    ricker_wavelet, gaussian_wavelet, ramp_wavelet, create_wavelet,
)

t, wav = ricker_wavelet(nt=500, dt=0.001, f0=25.0)
t, wav = gaussian_wavelet(nt=500, dt=0.001, f0=25.0)
t, wav = ramp_wavelet(nt=500, dt=0.001, t0=0.05)
t, wav = create_wavelet('ricker', nt=500, dt=0.001, f0=25.0)
```

## AVO Reflectivity

### Three Methods

```python
from geobrain.physics.wave import (
    Zoeppritz, AkiRichards, Shuey,
    compute_reflectivity, normal_incidence_rc,
)

# Exact solution
zoep = Zoeppritz()
rc = zoep(vp1=2000, vs1=1000, rho1=2.0,
          vp2=2500, vs2=1250, rho2=2.2,
          theta=[0, 15, 30, 45])

# Linearized approximation
aki = AkiRichards()
rc = aki(vp1, vs1, rho1, vp2, vs2, rho2, theta)

# Simplified (AVO attributes)
shuey = Shuey()
rc = shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta)
R0, G, F = shuey.avo_attributes(vp1, vs1, rho1, vp2, vs2, rho2)

# Factory function
rc = compute_reflectivity(..., method='zoeppritz')  # or 'aki', 'shuey'

# Normal incidence
R0 = normal_incidence_rc(vp1, rho1, vp2, rho2)
```

```{figure} ../../examples/figs/06_angle_gathers.png
:width: 100%
:name: fig-avo-angle-gathers

Synthetic AVO angle gathers from Shuey approximation.
```

## Convolution

```python
from geobrain.physics.wave import (
    convolve_trace, convolve_gather, apply_wavelet,
)

result = convolve_trace(trace, wavelet, mode='same')  # 'full', 'same', 'valid'
result = convolve_gather(gather, wavelet)              # All traces at once
result = apply_wavelet(reflectivity, wavelet)          # Matrix multiplication
```

## Boundary Conditions

```python
from geobrain.physics.wave import bc_pml, bc_gerjan, bc_sincos, create_boundary

damp = bc_pml(nx=200, nz=100, dx=10, dz=10, n_layers=20, vmax=3000)
damp = bc_gerjan(nx=200, nz=100, dx=10, dz=10, n_layers=20, alpha=0.005)
damp = bc_sincos(nx=200, nz=100, dx=10, dz=10, n_layers=20)

# Factory
damp = create_boundary('pml', nx=200, nz=100, dx=10, dz=10,
                        n_layers=20, vmax=3000)
```

## Seismic Data Management

```python
from geobrain.physics.wave import SeismicData

data = SeismicData(survey)
data.record(result)              # Store simulation output

pressure = data.get_pressure()              # (n_shots, nt, n_receivers)
pressure_norm = data.get_pressure(normalize=True)

data.save("seismic.npz")        # Persist to disk
data.load("seismic.npz")        # Load back

data_dict = data.to_dict()       # Get all components as dict
```

## Metrics

```python
from geobrain.physics.wave import mse, rmse, mape, snr

error = mse(true_model, inverted_model)
error = rmse(true_model, inverted_model)
error_pct = mape(true_model, inverted_model)
snr_db = snr(true_model, inverted_model)
```
