from robertastree.model import Tree

import os
import pandas as pd
import numpy as np
import torch

import pytest

# a toy classifier
if 'temporary' not in os.listdir():
    os.mkdir('./temporary')


class toyclassifier(torch.nn.Module):

    def __init__(self):

        super(toyclassifier, self).__init__()
        self.params = torch.nn.Linear(1, 1)

    def forward(self, sample):

        return torch.tensor([[0., 1.]], dtype=torch.float, requires_grad=True)


# dataset class
class toydataset(torch.utils.data.Dataset):

    def __init__(self, data):

        super(toydataset, self).__init__()
        self.features = data.drop(columns=['label'], axis=1)
        self.labels = data.label

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return {'sample': torch.tensor(self.features.iloc[idx], dtype=torch.float)},\
            torch.tensor(self.labels.iloc[idx])


def test_model_initiliazation():

    nclasses = 16

    # a toy trainset
    trainset = pd.DataFrame()
    trainset['feature'] = list(range(nclasses))
    trainset['label'] = list(range(nclasses))

    classifier = toyclassifier()

    model = Tree(classifier=classifier,
                 trainset=trainset,
                 validset=trainset.copy(),
                 models_path='./temporary/')

    assert model.n_classes == nclasses
    assert next(model.classifier.parameters()).is_cuda == torch.cuda.is_available()


def test_get_classifier_classes():

    nclasses = 4

    # a toy trainset
    trainset = pd.DataFrame()
    trainset['feature'] = list(range(nclasses))
    trainset['label'] = list(range(nclasses))

    classifier = toyclassifier()

    model = Tree(classifier=classifier,
                 trainset=trainset,
                 validset=trainset.copy(),
                 models_path='./temporary/')

    first_class = np.array([0, 1], dtype='int')
    second_class = np.array([2, 3], dtype='int')
    x, y = model._get_classifier_classes(0, 0)
    assert np.array_equal(x, first_class)
    assert np.array_equal(y, second_class)

    first_class = np.array([0], dtype='int')
    second_class = np.array([1], dtype='int')
    x, y = model._get_classifier_classes(1, 0)
    assert np.array_equal(x, first_class)
    assert np.array_equal(y, second_class)

    first_class = np.array([2], dtype='int')
    second_class = np.array([3], dtype='int')
    x, y = model._get_classifier_classes(1, 1)
    assert np.array_equal(x, first_class)
    assert np.array_equal(y, second_class)


def test_model():

    nclasses = 4

    # a toy trainset
    trainset = pd.DataFrame()
    trainset['feature'] = list(range(nclasses))
    trainset['label'] = list(range(nclasses))

    classifier = toyclassifier()

    model = Tree(classifier=classifier,
                 trainset=trainset,
                 validset=trainset.copy(),
                 models_path='./temporary/')

    with pytest.raises(Exception):
        model.test_classifier(0, 0, trainset.copy())

    model.configure(optimizer=torch.optim.SGD,
                    optimizer_params={'lr': 0.},
                    dataset_class=toydataset)

    model.train()

    accuracy = model.test_classifier(0, 0, trainset.copy())

    # the toy classifier always chose the second option
    # the samples are 4, and have labels 0,1,2,3
    # the expected accuracy of classifier [0,0] is 50.0%

    assert accuracy == 50.0
