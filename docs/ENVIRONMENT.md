# Environment

Use Python 3.12 and install `.[dev]` as described in the README. Dependencies
are declared in `pyproject.toml`.

| Component | Tested version |
|---|---|
| Python | 3.12.13 |
| NumPy | 2.5.2 |
| SciPy | 1.18.0 |
| PyTorch | 2.12.0+cu130 |
| pytest | 9.1.1 |

Primary training used PyTorch 2.12.0+cu130 with CUDA 13.0. The table describes
the checked local environment, not a complete historical lockfile. Fresh-machine
installation and exact EER reproduction have not yet been verified. Hardware
and library differences can affect reruns even with matching seeds.
