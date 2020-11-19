# QMC_LTFIM

An SSE QMC implementation of the quantum Ising model with transverse and/or longitudinal field

## TODOs

- 2-point ZZ correlators
  - structure factors
  - Need Lattices package:
    - 2-point correlators need access to the lattice struct in order to reduce redundant calculations
    - structure factors need to be able to do Fourier transforms over the lattice
    - we need to be able to translate site indices into site coordinates and take boundary conditions into account

- Revamp run script
  - Add "bin" option: kinda like "skip" but averages over K measurements to produce a single measurement
  - consider just making a separate script for ML data gen, since we don't need things like "skip" or saving of spin-configs
  - SIGTERM handling:
    - catch sigterm and checkpoint immediately
    - doesn't matter if we're in the middle of cluster construction, when we load the chain back up again we'll start with a full diagonal update anyway
    - could be bad if the sigterm happens in the middle of a cluster flip though...
      - maybe instead of "flipping" we "set" the legtype to a value that was sampled, that way we can just re-run the cluster_update! fn after loading up the checkpoint to make sure everything's consistent
      - wait but how do we know if it was interrupted during link list construction or the cluster update?

- Reduce Memory Usage
  - many places need this: Hamiltonian structs (use banded matrices for 1D for some savings), operator samplers, probability vectors
    - a "Hierarchical" Improved Operator Sampler probably isn't very helpful since it already stores O(N_b) data (the bonds themselves)
    - maybe the Operator Sampler could just sample a number, and then we'd use the inverse of the OperatorDict's "hash" to reconstruct the operator tuple, that way it won't have to store all the operator tuples.

- getweight/getlogweight should be the Hamiltonian's responsibility; may defer to op_sampler for complicated interactions but short circuiting for simple systems would be beneficial
  - probability vectors and op samplers should *optionally* store the weights

- Hamiltonian constructor given a Lattice
  - should be able to do Ewald summation over periodic boundaries (option)
  - Rydberg constructor

- Config files (draft idea, may not do all of this):
  - Base config will just state the Hamiltonian, and the lattice along with any relevant parameters
  - Config will be appended to by a utility script and will now contain a list of all (non-zero weight) operators
    - Bonds will have info on the two site they're acting on (both the site IDs and the coords) as well as the interaction strength
    - Site operators will have the site ID, site coords, site op type (sigma_x or sigma_z), and field strength
    - The coordinates and interaction strengths may ofc be modified by hand (though re-running the aforementioned utility script will overwrite those changes)
  - Config file will be read by the run script which will then take simulation parameters (temperature, initial operator list length, number of EQ/MC steps, what observables to record, etc.) before starting the run(s).

- Error Analysis:
  - Vanilla Jackknife/Bootstrap assume IID samples, and thus underestimate standard errors (this is true even for simple estimators which are just means). There are Bootstrap methods specifically for time-series data which could be useful.
    - There may be a way to turn one of these into a "streaming" bootstrap
  - Autocorrelation time we're computing seems to be larger than what's computed by other packages. Should try calculating autocorrelation time using a binning analysis instead as it may be more "stable" than the FFT based method we're using.
  - Should compute autocorrelation time for *all* observables in order to get accurate error bars.
