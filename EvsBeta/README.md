All calculations here were for 1D systems with periodic boundaries, J = 1, and 1 million samples drawn. The `energy_magnetization.txt` file's columns are structured as follows.

1. N: 1D chain length
2. \beta$: Inverse temperature
3. hx: transverse field strength
4. hz: parallel field strength
5. ED E/N: The energy obtained from ED (`LTFIM_thermal.jl`)
6. ED M/N: The magnetization obtained from ED
7. $\varepsilon$: The ED energy minus the QMC energy
8. $\pm$: The error in the QMC energy
9. $\theta$: The ED magnetization minus the QMC magnetization
10. $\pm$: The error in the QMC magnetization
