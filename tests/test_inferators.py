import torch
import numpy as np
from robertastree.inferators import WeightedAverageInferator, get_probabilities


def test_get_probabilities():
    '''
    positive unit test

    Verify that, if the outputs are all equal, the return probabilities
    are all equal.
    '''

    nclasses = 8
    nclassifiers = nclasses - 1

    tree_outputs = torch.ones((nclassifiers, 1, 2))

    probabilities = get_probabilities(tree_outputs)
    true_probabilities = torch.tensor([1. / nclasses for i in range(nclasses)])

    assert list(probabilities.keys()) == list(range(nclasses))

    assert probabilities[0].shape == (tree_outputs.shape[1],)

    for i in range(nclasses):
        assert probabilities[i] == true_probabilities[i]


def test_WeightedAverageInferator():
    '''
    positive unit test

    Verify the function predict the correct value for a 
    specific input.
    '''

    nclasses = 8
    nclassifiers = nclasses - 1

    tree_outputs = torch.ones((nclassifiers, 1, 2))

    probabilities = np.array([1. / nclasses for i in range(nclasses)])

    classes = dict([(i, (i, i + 1)) for i in range(8)])
    classes_midpoints = np.array([1 / 2 + i for i in range(nclasses)])

    weighted_average = (probabilities * classes_midpoints).sum()

    predicted_value = WeightedAverageInferator(tree_outputs, classes)

    assert predicted_value == weighted_average


def test_WeightedAverageInferator_output_shape():
    '''
    positive property test

    Verify the output has the correct shape.
    '''

    nclasses = 8
    nclassifiers = nclasses - 1
    batchsize = 10

    tree_outputs = torch.ones((nclassifiers, batchsize, 2))

    classes = dict([(i, (i, i + 1)) for i in range(8)])

    predicted_value = WeightedAverageInferator(tree_outputs, classes)

    assert predicted_value.shape == (batchsize,)
