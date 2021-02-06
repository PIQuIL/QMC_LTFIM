# QMC_LTFIM

An SSE QMC implementation of the quantum Ising model with transverse and/or longitudinal field

How to run TFIM/LTFIM (using multibranch cluster update):

```bash
cd scripts
# groundstate 10site periodic chain, projector length 1000, 100_000 samples
julia main.jl groundstate 10 -p -M 1000 -n 100000 -J 1.0 --hx 1.0 --hz 1.0

# thermalstate 8x4 square lattice PBC, projector length 2000, 100_000 samples, beta=20
julia main.jl mixedstate 8 4 -p -M 2000 -n 100000 -J 1.0 --hx 1.0 --hz 1.0 --beta 20.0
```
Currently (L)TFIM CLI only supports 2D square or 1D chain lattices with nearest-neighbor interactions

How to run r^-6 Rydberg (using line cluster update):
```bash
cd scripts
julia rydberg.jl groundstate 8 4 -M 4000 -n 100000 -R 1.2 --omega 1.0, --delta 1.0
```
Currently, for Rydberg on a 2D lattice boundary conditions are (open, periodic) and the `-p` flag is ignored.


- Misc Papers:
  - https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.165138
  - https://link.aps.org/doi/10.1103/PhysRevLett.101.210602
  - https://arxiv.org/pdf/1510.00979.pdf
  - https://arxiv.org/abs/1004.4577v2
  - https://arxiv.org/abs/1909.09652
