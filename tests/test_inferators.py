import torch
import numpy as np
from RobertasTree.inferators import WeightedAverageInferator


def test_WeightedAverageInferator():

    nclasses = 8
    nlayers = int(np.log2(nclasses))

    nclassifiers = nclasses - 1
    nclassifiers_from_layers = 0
    for i in range(nlayers):
        nclassifiers_from_layers += 2**i

    assert nclassifiers == nclassifiers_from_layers

    tree_outputs = np.ones((nclassifiers, 2))

    probabilities = [1. / nclasses for i in range(nclasses)]

    classes = dict([(i, (i, i + 1)) for i in range(8)])
    classes_midpoints = np.array([1 / 2 + i for i in range(nclasses)])

    weighted_average = probabilities * classes_midpoints

    predicted_value = WeightedAverageInferator(tree_outputs, classes)

    assert predicted_value == weighted_average
