# QMC_LTFIM
An SSE QMC implementation of the quantum Ising model with transverse and/or longitudinal field

## Status

The current code runs and gives accurate results for PBC but, we need to
test/debug OBC better: ED values are currently outside of error bars, but I'm
using a single stdev of the mean as error bars, so we should construct 95%
CI's to make a better comparison.
