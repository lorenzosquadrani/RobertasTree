import torch
import numpy as np
from robertastree.inferators import WeightedAverageInferator, get_probabilities


def test_get_probabilities():
    '''
    positive unit test

    Verify that, for a given tree output, correct probabilities are given.
    '''

    tree_outputs = torch.tensor([[
        [0.2, 0.8],
        [0.4, 0.6],
        [0.12, 0.88]
    ]])

    probabilities = get_probabilities(tree_outputs)

    normalized_outputs = torch.nn.functional.softmax(tree_outputs, dim=2)

    true_probabilities = torch.tensor([
        [normalized_outputs[0, 0, 0] * normalized_outputs[0, 1, 0],
         normalized_outputs[0, 0, 0] * normalized_outputs[0, 1, 1],
         normalized_outputs[0, 0, 1] * normalized_outputs[0, 2, 0],
         normalized_outputs[0, 0, 1] * normalized_outputs[0, 2, 1]]
    ])

    assert torch.equal(probabilities, true_probabilities)


def test_WeightedAverageInferator():
    '''
    positive unit test

    Verify the function predict the correct value for a
    specific input.
    '''

    nclasses = 8
    nclassifiers = nclasses - 1
    batchsize = 1

    # make outputs of the tree, all equals
    tree_outputs = torch.ones((batchsize, nclassifiers, 2))

    # the expected probabilities are all equal
    probabilities = np.array([1. / nclasses for i in range(nclasses)])

    # some fake classes
    classes = dict([(i, (i, i + 1)) for i in range(8)])
    classes_midpoints = np.array([1 / 2 + i for i in range(nclasses)])

    # the expected weighted average
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

    tree_outputs = torch.ones((batchsize, nclassifiers, 2))

    classes = dict([(i, (i, i + 1)) for i in range(8)])

    predicted_value = WeightedAverageInferator(tree_outputs, classes)

    assert predicted_value.shape == (batchsize,)
