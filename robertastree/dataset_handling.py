import numpy as np
import pandas as pd


def get_subdatasets(dataset, i, j, random_state=0):
    '''
    From the dataset with all samples, extract the samples on which classifier i,j should be trained.
    Samples are already organized in two classes, with labels 0 and 1.

    Parameters
    ---------------------
    dataset : pandas.DataFrame
        The dataset containing all the samples with labels. The labels' column must be named 'label'.

    i : int

    j : int

    random_state : int


    Return
    ----------------------

    new_dataset : pd.Dataframe
    '''

    criteria1, criteria2 = get_criteria(dataset, i, j)

    subdataset1 = dataset[criteria1].copy()
    subdataset2 = dataset[criteria2].copy()

    subdataset1["label"] = 0
    subdataset2["label"] = 1

    new_dataset = pd.concat([subdataset1, subdataset2], axis=0).sample(frac=1)

    return new_dataset


def get_criteria(dataset, i, j):
    '''
    Produce two boolean 1D numpy array, which can be used to select the
    samples beloging to the two classes of classifier i,j of a
    Roberta's tree.

    Parameters
    ----------------------
    dataset : pandas.DataFrame
        The dataset from which samples will be selected. Must have a column named 'label'.
        The number of different labels must be a power of 2.

    i : int

    j : int


    Returns
    -----------------------
    criteria1 : numpy.array

    criteria2 : numpy.array
    '''

    classes = dataset.label.unique()
    classes.sort()

    possible_classes = np.split(classes, 2**(i + 1))

    first_class, second_class = possible_classes[2 * j], possible_classes[2 * j + 1]

    criteria1 = np.full(len(dataset), False)
    criteria2 = np.full(len(dataset), False)
    for label1, label2 in zip(first_class, second_class):
        criteria1 = criteria1 | (dataset["label"] == label1).values
        criteria2 = criteria2 | (dataset["label"] == label2).values

    return criteria1, criteria2


def balance_dataset(dataset):
    '''
    Receive a dataset with labelled samples and balance it.
    The mean number of samples per label, N, is computed.
    Over-sampling and under-sampling are performed until
    the number of samples per each label is identically equal to N.

    Parameters
    ----------
      dataset : pandas.DataFrame
        The dataset to balance. Must contain a column named "label".

    Returns
    -------
      balanced_dataset : pandas.DataFrame
        The balanced version of the initial dataset. Samples are shuffled.
    '''
    counts = list(dataset.label.value_counts())
    mean_count = sum(counts) // len(counts)

    subdata = {}

    for label in dataset.label.unique():
        subdata[label] = dataset[dataset.label == label]

    balanced_dataset = pd.concat(
        [subdata[label].sample(mean_count, replace=True) for label in subdata],
        axis=0)

    return balanced_dataset.sample(frac=1).reset_index()


def from_range_to_classes(targets, n_classes, value_range=None):
    '''
    Divide a numerical range in a given number of intervals, and gives the
    correspondent labels to a Series of numerical values.

    Parameters
    ----------
      targets : pandas.Series
        Targets of the training samples

      n_classes : int
        Number of desired classes

      value_range : float tuple
        The range to divide in classes. If None, the minimum and maximum value
        from targets are used.

    Returns
    -------
      labels : pandas.Series
        The series containing labels of the training samples in place of the
        targets

      classes : dict
        The classes' labels (keys of the dict) with the correspondent numerical
        range (as a float tuple)
    '''

    if value_range is None:
        value_range = (targets.min(), targets.max())

    x = np.linspace(value_range[0], value_range[1], n_classes + 1)
    classes = {}
    for i in range(n_classes):
        classes[str(i)] = (x[i], x[i + 1])

    def _get_class(value):
        for key in classes:
            if (value >= classes[key][0] and value <= classes[key][1]):
                return int(key)
        raise ValueError("Find a value out of range! I don't feel like going on...")

    labels = targets.apply(_get_class)

    return labels, classes
