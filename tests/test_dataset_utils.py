# Tests for RobertasTree.dataset_handling module

from RobertasTree.dataset_handling import from_range_to_classes

import pandas as pd
import numpy as np


# Testing function from_range_to_classes

def test_correct_classes_and_labels():
    '''
    Positive unit test.
    '''

    targets = pd.Series(np.arange(5, 100, 10))
    n_classes = 10
    value_range = (0, 100)

    labels, classes = from_range_to_classes(targets=targets,
                                            n_classes=n_classes,
                                            value_range=value_range)

    assert len(classes.keys) == 10

    for key in classes:
        assert classes[key][1] - classes[key][0] == 10

    for n in labels.value_counts():
        assert n == 1


def test_correct_classes_number():
    '''
    Positive property test.

    Verify that the returned dict 'classes' containes the number
    of classes specified in the argument 'n_classes'.
    '''

    targets = pd.Series(np.random.random(100))
    n_classes = 10
    _, classes = from_range_to_classes(targets=targets, n_classes=n_classes)
    assert n_classes == len(classes.keys())
