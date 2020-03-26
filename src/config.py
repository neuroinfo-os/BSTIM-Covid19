import itertools as it

diseases = ["covid19"]
prediction_regions = ["germany"]

combinations_age_eastwest = [(False,False),(False,True),(True,True)]
combinations = list(it.product(range(len(combinations_age_eastwest)), diseases))
