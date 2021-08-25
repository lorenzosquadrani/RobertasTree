# Tests for robertastree.dataset_handling module

from robertastree.dataset_handling import (
    from_range_to_classes,
    balance_dataset,
    get_criteria,
    get_subdatasets
)

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

    assert len(classes.keys()) == 10

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


# Testing function balance_dataset

def test_final_dataset_is_balanced():
    '''
    Positive unit test.

    Check that the return dataset is now balanced,
    and the value coincide with the initial mean
    value of samples per label.
    '''

    dataset = pd.DataFrame()

    dataset['label'] = pd.Series(np.random.randint(0, 10, 500))
    counts = dataset.label.value_counts()
    mean_count = sum(counts) // len(counts)

    balanced_dataset = balance_dataset(dataset)

    for x in balanced_dataset.label.value_counts():
        assert x == mean_count


# Tests for function get_criteria

def test_correct_criteria():
    '''
    Positive unit test.

    Verify that, for a toy example, the right number
    of samples are selected for each classifier.
    '''

    dataset = pd.DataFrame({"label": [0, 1, 2, 3, 4, 5, 6, 7]})

    n_layers = 3

    for i in range(n_layers):
        for j in range(2**i):
            criteria1, criteria2 = get_criteria(dataset, i, j)

            assert criteria1.sum() == 8 / 2**(i + 1)
            assert criteria2.sum() == 8 / 2**(i + 1)


def test_correct_subdatasets():
    '''
    Positive unit test.


    '''

    # create a toy dataset with labels 0,1,2,3 and number of samples n0,n1,n2,n3
    toy_dataset = pd.DataFrame()

    n0 = 5
    n1 = 12
    n2 = 14
    n3 = 19

    toy_dataset['label'] = [0, ] * n0 + [1, ] * n1 + [2, ] * n2 + [3, ] * n3

    # the number of classifiers for 4 classes is 4-1 = 3, identified by
    # (i=0, j=0), (i=1, j=0), (i=1,j=1)

    # without test fraction
    sub0_0 = get_subdatasets(toy_dataset, 0, 0)
    sub1_0 = get_subdatasets(toy_dataset, 1, 0)
    sub1_1 = get_subdatasets(toy_dataset, 1, 1)

    assert len(sub0_0) == (n0 + n1 + n2 + n3)
    assert len(sub1_0) == (n0 + n1)
    assert len(sub1_1) == (n2 + n3)
