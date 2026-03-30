# Liang-Kleeman information flow (LKIF)

This repository provides the codes for computing the rate of information transfer (in other words, the causal influences) between variables:
- function_liang.py allows to compute the rate of information transfer between 2 variables based on [Liang (2014)](https://doi.org/10.1103/PhysRevE.90.052150)
- function_liang_nvar.py allows to compute the rate of information transfer for multivariate time series based on [Liang (2021)](https://doi.org/10.3390/e23060679)
- function_liang_nvar_adapted.py is an extension of the previous code with the possibility to choose between two types of error computation.

If you use one of these codes, please cite one of the following papers:
- Docquier, D., S. Vannitsem, F. Ragone, K. Wyser, X. S. Liang (2022). Causal links between Arctic sea ice and its potential drivers based on the rate of information transfer. _Geophysical Research Letters_, [https://doi.org/10.1029/2021GL095892](https://doi.org/10.1029/2021GL095892).
- Docquier, D., G. Di Capua, R. V. Donner, C. A. L. Pires, A. Simon, S. Vannitsem (2024). A comparison of two causal methods in the context of climate analyses, _Nonlinear Processes in Geophysics_, [https://doi.org/10.5194/npg-31-115-2024][(https://doi.org/10.5194/npg-31-115-2024).

Codes developed by [David Docquier](https://climdyn.meteo.be/team/david-docquier) (RMI).
