from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pandas as pd


class RobertasTreeDatasetForInference(Dataset):
    def __init__(self, dataset):

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        self.robertas_data = self.tokenizer(dataset.excerpt.tolist(),
                                            padding=True,
                                            truncation=True,
                                            return_tensors='pt')

        self.ids = dataset.id.tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        attention_mask = torch.tensor(self.robertas_data['attention_mask'][idx])
        input_ids = torch.tensor(self.robertas_data['input_ids'][idx])
        ids = self.ids[idx]

        return attention_mask, input_ids, ids


class CommonLitDataset(Dataset):
    '''
    Custom Pytorch dataset for text classification.

    Parameters
    ----------
        data : pandas.Dataframe
            Must have a columns named 'excerpt', containing texts to be
            classified, and a column named 'label', containing corresponding
            labels.

        tokenizer : Huggingface Tokenizer

        max_len : int
            Maximum number of tokens for a text sample. All samples will be
            padded/truncated to it.
    '''

    def __init__(self, data):

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.max_len = 300

        self.excerpts = data.excerpt.values.tolist()
        self.labels = data.label.values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        excerpt, labels = self.excerpts[idx], self.labels[idx]
        features = self.tokenize(excerpt)

        samples = {'input_ids': torch.tensor(features['input_ids'], dtype=torch.long),
                   'attention_mask': torch.tensor(features['attention_mask'], dtype=torch.long)}
        labels = torch.tensor(labels, dtype=torch.long)

        return samples, labels

    def tokenize(self, data):
        data = data.replace('\n', '')

        tok = self.tokenizer.encode_plus(
            data,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,
        )

        padding_length = self.max_len - len(tok['input_ids'])
        tok['input_ids'] = tok['input_ids'] + ([0] * padding_length)
        tok['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)
        return tok


def get_subdatasets(dataset, i, j, test_frac=0., random_state=0):
    '''
    From the dataset with all samples, extract the samples on which classifier i,j should be trained.
    Samples are already organized in two classes, with labels 0 and 1.

    Parameters
    ---------------------
    dataset : pandas.DataFrame
        The dataset containing all the samples with labels. The labels' column must be named 'label'.

    i : int

    j : int

    test_frac : float

    random_state : int


    Return
    ----------------------

    if test_frac != 0 :  (pd.Dataframe, pd.Dataframe)

    else              :   pd.Dataframe
    '''

    criteria1, criteria2 = get_criteria(dataset, i, j)

    subdataset1 = dataset[criteria1].copy()
    subdataset2 = dataset[criteria2].copy()

    subdataset1["label"] = 0
    subdataset2["label"] = 1

    new_dataset = pd.concat([subdataset1, subdataset2], axis=0).sample(frac=1)

    if test_frac != 0:
        trainset, testset = train_test_split(new_dataset,
                                             test_size=test_frac,
                                             random_state=random_state,
                                             stratify=new_dataset.label)
        return trainset, testset
    else:
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

    def get_class(value):
        for key in classes:
            if (value >= classes[key][0] and value <= classes[key][1]):
                return int(key)
        raise ValueError("Find a value out of range! I don't feel like going on...")

    labels = targets.apply(get_class)

    return labels, classes
