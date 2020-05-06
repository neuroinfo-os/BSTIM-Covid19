import itertools as it

""" define all importamt meta variables and product lists of experiments """

# TODO:
#      * Interaction effect
#      * Age (Demographic) as predictor
#      * delay polynomial
#      * LATER: dates for end of training start of testing

diseases = ["covid19"]
prediction_regions = ["germany"]

combinations_ia_report = [(False,False), (False,True), (True,False), (True,True)]


combinations = list(it.product(
    range(len(combinations_ia_report)), diseases))
