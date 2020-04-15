import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib as mpl
import json
import csv
import random
import timeit
import re

from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from descartes import PolygonPatch
from matplotlib import pyplot as plt
from collections import OrderedDict

from geo_utils import *


def merge_age_groups(age_group):
    pat = re.compile("A([0-9]+).*")
    starting_age = int(pat.match(age_group).group(1))

    if starting_age < 5:
        return "[0-5)"
    elif starting_age < 20:
        return "[5-20)"
    elif starting_age < 65:
        return "[20-65)"
    else:
        return ">65"


def county_shapes(shape_data_path="../data/raw/germany_county_shapes.json"):
    """ extract county shapes """

    counties = OrderedDict()
    with open(shape_data_path, 'r') as data_file:
        shape_data = json.load(data_file)

    for val in shape_data["features"]:
        id_current = val["properties"]["RKI_ID"]
        name_current = val["properties"]["RKI_NameDE"]

        if val["geometry"]["type"] == "Polygon":
            polygon_current = Polygon(
                val["geometry"]["coordinates"][0], val["geometry"]["coordinates"][1:])
        elif val["geometry"]["type"] == "MultiPolygon":
            polys = [Polygon(p[0], p[1:])
                     for p in val["geometry"]["coordinates"]]
            polygon_current = MultiPolygon(polys)

        counties[id_current] = {"shape": polygon_current, "name": name_current}

    assert counties["03404"]["name"] == "SK Osnabrück"

    return counties


def add_demographic_information(
        counties,
        demographic_data_path="../data/raw/germany_population_data.csv"):

    county_names = dict([(val["name"], id) for id, val in counties.items()])
    age_data = pd.read_csv("../data/raw/germany_population_data.csv")[
        ["county", "age_group", "year", "population"]]

    age_data["age_group"] = age_data["age_group"].apply(merge_age_groups)
    age_data["id"] = age_data["county"].apply(county_names.get)
    total_population_data = age_data.groupby(
        ["id", "year"]).aggregate({"population": 'sum'})
    group_population_data = age_data.groupby(
        ["id", "age_group", "year"]).aggregate({"population": 'sum'})
    log_group_fraction_data = group_population_data.apply(lambda row: np.log(
        row / total_population_data.loc[(row.name[0], row.name[2])]), axis=1)
    total_population_data["age_group"] = "total"
    total_population_data.reset_index(inplace=True)
    total_population_data.set_index(["id", "age_group", "year"], inplace=True)
    population_data = pd.concat(
        [log_group_fraction_data, total_population_data])

    for id, row in population_data.unstack(1).unstack(1).iterrows():
        counties[id]["demographics"] = row["population"].to_dict()


def add_testpoint(counties, num_testpoints_per_county=500):

    for i, (id, county) in enumerate(counties.items()):
        print("Sampling testpoints for county '{}' ({:.2f}%)".format(
            id, 100.0 * i / len(counties)), end="\r")
        # Sample uniformly in the region using local jacobian
        centroid = np.array(county["shape"].centroid)
        Σ_sq = jacobian_sq(centroid[1])
        Σ_sqinv = np.diag(1 / np.diag(Σ_sq))

        x1, y1, x2, y2 = county["shape"].bounds
        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])

        p1_corrected = (p1 - centroid).dot(Σ_sq)
        p2_corrected = (p2 - centroid).dot(Σ_sq)

        dims = p2_corrected - p1_corrected

        # calculate testpoints
        testpoints = np.empty((num_testpoints_per_county, 2), np.float)

        cnt = 0
        while cnt < num_testpoints_per_county:
            tp = (np.random.rand(2) * dims).dot(Σ_sqinv) + p1
            if not county["shape"].contains(Point(tp)):
                continue
            testpoints[cnt, :] = tp
            cnt += 1

            counties[id]["testpoints"] = testpoints


def add_regions(counties):

    regions = OrderedDict({
        "germany": {"ids": list(counties.keys()), "hatch": "+"},
        "bavaria": {"ids": list(filter(lambda x: x.startswith("09"), counties.keys())), "hatch": "\\"},
        "east": {"ids": list(filter(lambda x: x[:2] > "11", counties.keys())), "hatch": "/"},
        "berlin": {"ids": list(filter(lambda x: x.startswith("11"), counties.keys())), "hatch": "o"}
    })

    for county in counties.values():
        county["region"] = []

    for region, props in regions.items():
        for id in props["ids"]:
            counties[id]["region"].append(region)

    for _, region in regions.items():
        region_poly = Polygon()
        for c in region["ids"]:
            region_poly = region_poly.union(counties[c]["shape"])

        region_poly = region_poly.buffer(0.01).simplify(0.01).buffer(-0.01)

        region["shape"] = region_poly

    return regions


def add_border_effects(counties, regions, sigmas, num_partition_samples=500):

    eps = np.random.randn(num_partition_samples, 2)

    for i, (id, county) in enumerate(counties.items()):
        county["borderloss"] = {}
        for j, (name, region) in enumerate(regions.items()):
            if id not in region["ids"]:
                continue

            print("Calculating border effects for county '{}' in region '{}' ({:.2f}%)".format(
                id, name, 100.0 * ((i * len(regions) + j) / (len(counties) * len(regions)))), end="\r")
            county["borderloss"][name] = compute_loss(
                region["shape"], county["testpoints"], sigmas, eps)


def county_county_interactions(counties, sigmas):
    factor = np.empty((len(counties), len(counties), len(sigmas)), np.float64)

    for i, (id_1, county_1) in enumerate(counties.items()):
        for j, (id_2, county_2) in enumerate(counties.items()):
            if j < i:
                factor[i, j, :] = factor[j, i, :]
                continue
            print("Calculating interaction effects for counties '{}' and '{}' ({:.2f}%)".format(
                id_1, id_2, 100.0 * (i + j / len(counties)) / len(counties)), end="\r")
            factor[i, j, :] = compute_interaction(
                county_1["testpoints"], county_2["testpoints"], sigmas)

    return factor


if __name__ == "__main__":
    """ TODO: add argument parser in the future """

    # NOTE: this is specific to the mode we are using!
    num_interaction_gaussians = 4
    interaction_distance = 50.0  # km
    sigmas = 2**np.arange(num_interaction_gaussians) / \
        (2 * num_interaction_gaussians) * interaction_distance

    num_testpoints_per_county = 500
    num_partition_samples = 500

    counties = county_shapes()
    add_demographic_information(counties)
    add_testpoint(counties)

    regions = add_regions(counties)
    add_border_effects(counties, regions, sigmas)

    factor = county_county_interactions(counties, sigmas)

    with open('../data/counties/counties.pkl', "wb") as f:
        pkl.dump(counties, f)

    np.save("../data/counties/interaction_effects.npy", factor)
