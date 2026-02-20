# Tutorial 15: Joint Seismic-Electrical Inversion (Sleipner)

```{note}
This tutorial is available as a Python script [`examples/15_sleipner_joint_inv.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/15_sleipner_joint_inv.py) and an interactive Jupyter notebook [`examples/notebooks/15_sleipner_joint_inv.ipynb`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/notebooks/15_sleipner_joint_inv.ipynb).
```

Jointly invert seismic and resistivity data to recover porosity and CO&#8322;
saturation for the Sleipner CO&#8322; storage site in the Norwegian Sea.

## What You Will Learn

- Combine multiple physics (seismic + resistivity) in a single objective
- Use rock physics to link seismic velocities and electrical resistivity
- Build a 3D convolutional decoder for network parameterization
- Run joint inversion with `Inverter.create_network()`
- Interpret CO&#8322; saturation from inverted properties

## Key Concepts

**Joint inversion** exploits complementary sensitivities of different
geophysical methods. Seismic data are sensitive to velocity contrasts while
resistivity data (via Archie's law) are sensitive to fluid saturation
changes caused by CO&#8322; replacing brine.

**Network parameterization** uses a 3D convolutional decoder as the model
representation, providing implicit spatial regularization and enabling
joint inversion of multiple properties from a shared latent space.

The Sleipner CO&#8322; storage project has injected CO&#8322; into the Utsira
Sand since 1996, making it one of the longest-running industrial carbon
storage operations. Time-lapse seismic monitoring has tracked the growing
CO&#8322; plume, and complementary resistivity data provide additional
sensitivity to saturation changes. This tutorial combines both data types in
a single objective function, using rock-physics relationships (Soft Sand,
Gassmann fluid substitution, and Archie's law) to link the shared model
parameters -- porosity and CO&#8322; saturation -- to the two sets of
observations.

## Code

```python
from geobrain.physics.wave import Shuey, RickerWavelet, create_conv_matrix
from geobrain.physics.rock import SoftSand, Gassmann, ArchieResistivity
from geobrain.optim import Inverter

inverter = Inverter.create_network(
    decoder=decoder,
    forward_fn=joint_forward,
    observed=d_obs,
)

result = inverter.run(max_epochs=1000, lr=0.001)
```

## Results

The observed data comprise seismic traces and resistivity measurements, both
of which carry information about the subsurface porosity and CO&#8322;
saturation distribution.

```{figure} ../../examples/figs/15_observed_data.png
:width: 100%
:name: fig-15-observed-data

Observed seismic and resistivity data.
```

The joint inversion recovers porosity and CO&#8322; saturation models that
are consistent with both data types simultaneously. The network
parameterization enforces spatial smoothness and couples the two properties
through a shared latent code.

```{figure} ../../examples/figs/15_inverted_models.png
:width: 100%
:name: fig-15-inverted-models

Inverted porosity and CO<sub>2</sub> saturation models.
```

The convergence curve shows steady reduction of the combined objective
function over iterations, confirming that the optimizer successfully minimizes
the misfit to both seismic and resistivity data.

```{figure} ../../examples/figs/15_convergence.png
:width: 100%
:name: fig-15-convergence

Joint inversion convergence curve.
```

## Full Example

See [`examples/15_sleipner_joint_inv.py`](https://github.com/GeoBrain-Project/geobrain/blob/main/examples/15_sleipner_joint_inv.py).
