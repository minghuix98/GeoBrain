# Tutorial 06: Seismic AVO Forward Modeling

```{note}
This tutorial is available as a Python script [`examples/06_seismic_avo_forward.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/06_seismic_avo_forward.py) and an interactive Jupyter notebook [`examples/notebooks/06_seismic_avo_forward.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/06_seismic_avo_forward.ipynb).
```

Generate pre-stack seismic angle gathers from geological models through rock
physics and AVO reflectivity.

## What You Will Learn

- End-to-end workflow: porosity &rarr; velocities &rarr; reflectivity &rarr; seismic
- Use `compute_reflectivity` for angle-dependent AVO modeling
- Build convolution matrices with `create_conv_matrix`
- Generate synthetic pre-stack angle gathers

## Key Concepts

AVO (Amplitude Variation with Offset) analysis exploits the fact that
reflection amplitudes change with incidence angle, providing information
about fluid content and lithology.

## Code Highlights

```python
from geobrain.physics.wave import RickerWavelet, compute_reflectivity, create_conv_matrix

# Angle-dependent reflectivity
refl = compute_reflectivity(vp1, vs1, rho1, vp2, vs2, rho2,
                            theta=[12, 24, 36], method='shuey')

# Wavelet convolution
ricker = RickerWavelet()
wavelet, _ = ricker(f0=25.0, dt=0.001)
W = create_conv_matrix(wavelet, n_samples)
seismic = refl @ W[nsW:-nsW].T
```

## Results

```{figure} ../../examples/figs/06_ricker_wavelets.png
:width: 100%
:name: fig-06-ricker-wavelets

Ricker wavelets at different dominant frequencies.
```

```{figure} ../../examples/figs/06_rock_properties.png
:width: 100%
:name: fig-06-rock-properties

Elastic property fields: Vp, Vs, density.
```

```{figure} ../../examples/figs/06_angle_gathers.png
:width: 100%
:name: fig-06-angle-gathers

Synthetic pre-stack seismic angle gathers.
```

## Full Example

See [`examples/06_seismic_avo_forward.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/06_seismic_avo_forward.py).
