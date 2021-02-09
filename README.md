# QMC_LTFIM

An SSE QMC implementation of the quantum Ising model with transverse and/or longitudinal field

How to run TFIM/LTFIM (uses the multibranch cluster update):

```bash
cd scripts
# groundstate 10site periodic chain, projector length 1000, 100_000 samples
julia main.jl groundstate 10 -p -M 1000 -n 100000 -J 1.0 --hx 1.0 --hz 1.0

# thermalstate 8x4 square lattice PBC, projector length 2000, 100_000 samples, beta=20
julia main.jl mixedstate 8 4 -p -M 2000 -n 100000 -J 1.0 --hx 1.0 --hz 1.0 --beta 20.0
```

Note: passing `--hz 0.0` will actually use the more general LTFIM cluster update algorithm,
which is slightly less efficient than the TFIM implementation. If you want to simulate a
TFIM model, either pass `--hz nothing` or don't use the `--hz` option at all.
Currently (L)TFIM CLI only supports 2D square or 1D chain lattices with nearest-neighbor interactions.
However, you may provide an interaction matrix to the simulation directly like so:

```bash
julia main.jl groundstate 10 -M 1000 -n 100000 -J "./path/to/file" --hx 0.7 --hz 0.2
```

Note that in this case, the interaction matrix's data overrides some other arguments. For example,
the number of sites passed to the CLI, and the `-p` flag will be ignored.
The format of the interaction matrix must be a text file in the format outputted by the `DelimitedFiles`
Julia package. Additionally, it is assumed that `J_ij` is a strictly upper triangular matrix;
any diagonal or sub-diagonal entries will be ignored. Be warned that infinite entries will
cause the simulation to crash.

How to run r^-6 Rydberg (uses the line cluster update):

```bash
julia rydberg.jl groundstate 8 4 -M 4000 -n 100000 -R 1.2 --omega 1.0 --delta 1.0
```

Currently, for Rydberg on a 2D lattice boundary conditions are (open, periodic) and the `-p` flag is ignored.


Both scripts allow passing the flag `--runstats` which computes simulation statistics such as diagonal/cluster update acceptance rates, cluster sizes, cluster abortion rates (only for the line update), etc. This flag is currently only supported for groundstate simulations.


- Misc Papers:
  - https://journals.aps.org/prb/abstract/10.1103/PhysRevB.88.165138
  - https://link.aps.org/doi/10.1103/PhysRevLett.101.210602
  - https://arxiv.org/pdf/1510.00979.pdf
  - https://arxiv.org/abs/1004.4577v2
  - https://arxiv.org/abs/1909.09652
