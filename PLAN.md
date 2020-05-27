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
        --> convert to new .csv src/preprocess_covid19_table.py (can use current .csv available 
            in /data/diseases)
    1. sample_ia_effects.py
    1. sample_posterior.py (sample_predictions is now subsumed in sample_posterior)
    1. model_selection.py
    1. Plots: (via command-line calls to pdf; or via notebooks/visualization)
        (curves, curves_appendix, interaction_kernel, temporal_contribution)

* "Final Run"
    1. 1 Model (id = 15 in combinations -> max parameters)
    1. results:
        * parameter trace
        * prediction on training & test set for plotting
            * no nowcast; 5 days from now for train cut-off/ forecast begin. 5 days into the future.
            * plots?
            * deviance / DS score distributions!


* Plots
    1. 1A qudaratisch; 1B 1C + 2?!--> appendix
    1. mean trend + periodic // no confidence intervals because VERY LARGE VARIANCE (that seems correlated)
    1. deviance/DS score into appendix distribution outlier? ("interpretation")
    1. Main fig: 2 cities full model \\ -5W(nowcast) -> -5D (forecast) -> +5D (forecast forecast)
    
