## Models done before hand-off

* Priority 1: Forecasting
    * 1 IA Kernel + Delay Polynom 1 day ahead
    * Deviance on test set -- only on the last day // for all counties.
    * WAIC on training set (model_selection.py) // model comparison: with / without IA Kernel | with / without Delay Polynom
    * Run WAIC comparions on Jureca

* Priority 2: Model Extensions
    * IA Kernel with limited temporal support & different sets of weights.

* Dashboard:
    * temporal evolution of deviance per county deviance for growing test set must be implemented
