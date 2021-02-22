# DMRG for 1D Rydberg chains

### Requirements:

- [ITensor (C++)](https://github.com/ITensor/ITensor)

### Run a DMRG calculation from command line:

```
$ make clean
$ make main
$ ./main N Rb delta Omega trunc num_samples
```

##### Positional argument descriptions:

- `N`: 1D chain length
- `Rb`: Blockade radius
- `delta`: detuning
- `Omega`: Rabi frequency
- `trunc`: maximum range interaction truncation (i.e. trunc=3 means third NN)
- `num_samples`: number of samples to generate in Sz basis

##### Output

- Samples in Sz basis go in `samples/DMRG_samples_N=<N>_Rb=<Rb>_delta=<delta>_Omega=<Omega>_trunc=<trunc>`
- Energy and magnetization go in `observables/DMRG_observables_N=<N>_Rb=<Rb>_delta=<delta>_Omega=<Omega>_trunc=<trunc>`
