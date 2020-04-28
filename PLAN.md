## Models done before hand-off

* Priority 1: Forecasting
    * 1 IA Kernel + Delay Polynom 1 day ahead
    * Deviance on test set -- only on the last day // for all counties.
    * WAIC on training set (model_selection.py) // model comparison: with / without IA Kernel | with / without Delay Polynom
    * Run WAIC comparions on Jureca

* Priority 2: Model Extensions
    * IA Kernel with limited temporal support & different sets of weights.
    * rm delay polynomial from model for predictions

* Dashboard:
    * temporal evolution of deviance per county deviance for growing test set must be implemented


* Workflow (current -- keep up to date):
    1. download current .csv database (see Readme, manual atm) 
        --> convert to new .csv src/preprocess_covid19_table.py
    1. sample_ia_effects.py
    1. sample_posterior.py (sample_predictions is now subsumed in sample_posterior)
    1. Plots: (via command-line calls to pdf; or via notebooks/visualization)
        (curves, curves_appendix, interaction_kernel, temporal_contribution)
