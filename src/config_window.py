import itertools as it


""" setup for model selection experiment """

# diseases = ["covid19"]
# prediction_regions = ["germany"]

ia_samples = list(range(100))
start = list(range(17,137))


combinations = list(it.product(ia_samples, start))





