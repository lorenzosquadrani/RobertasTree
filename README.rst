.. raw:: html

   <a href="https://www.google.com/search?q=what+you+can+find+by+clicking+on+a+pear">
      <img alt="A pear" width="100px" src="https://icons.iconarchive.com/icons/alex-t/fresh-fruit/256/pear-icon.png" align="right">
   </a> 
   
.. list-table::
   :header-rows: 1

   * - Author
     - Documentation
     - Codacy
   * - `L. Squadrani <https://github.com/lorenzosquadrani>`_
     - TO DO
     - |quality badge| |coverage badge|


   
############
RobertasTree
############   

RobertasTree is a framework to convert a Pytorch multi-class classifier into a tree-like deep learning model with better perfomances.

.. contents:: Table of Contents:
   :local:


Overview
========   
   
The idea behind RobertasTree is very simple and general.
Consider a N-classes classification task, where N is a power of 2.
This task can be decomposed in N-1 binary classification tasks, organized in a
tree-like structure like in figure.
Each node of the tree correspond to a classifier trained to distinguish between progressively more specific classes: the first node decides if the input belongs to class {0,...,n} or {n+1,...,N}, the first node in the second layer decides if the input belongs to class {0,...,d} or {d+1,...,n}, and so on.

Such approach could be advantageous or disadvantageous according to your data. 
In general, the advantages that you may get are the following (not verified, just guessing):

-  as the same data are utilized to train different classifier on different binary class, a certain form of data augmentation is obtained.
-  the number of parameters of the model are increased, without increasing overfitting risk (I hope).

RobertasTree was born as a deep learning model to compete in the Kaggle competition `CommonLitReadibility <https://www.kaggle.com/c/commonlitreadabilityprize>`_.
The first implementation was thus a task-specific, and was based on Roberta transformer (ref).
While the name remains RobertasTree, here we have tried to develop a general framework: custom classifier, dataset, training algorithm are allowed. 
Of course, the main limitation is unchanged: the number of classes must be a power of 2.


Since the CommonLitReadibility competition was a regression task, RobertasTree framework also inherited functions to covert a regression task in a classification tasks, and backward. 
This first conversin is simply done by subdividing the target range in intervals and assigning a class labels to them.
The classification results are then converted into regression results by a weighted average of the intervals' mid values (where the weights are the predicted classes' probabilities).
(see Usage section for clearness).


Prerequisites
=============

-  numpy
-  pandas
-  torch

Installation
============

The installation steps are

.. code:: bash

   python -m pip install -r ./requirements.txt

to install the prerequisites and then

.. code:: bash

   python setup.py install

or for installing in development mode:

.. code:: bash

   python setup.py develop --user

Usage
=====

Classification
--------------

Here there is the pipeline to apply RobertasTree on MNIST (ref) (I do not recommend to try it
on your easy-running-out-of-memory laptop. Use Google Colab instead.).

First we prepare the dataset.

.. code:: python

   # Download dataset
   from sklearn.datasets import fetch_openml
   X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

   # Select a power 2 number of classes
   X = X[y < 8]
   y = y[y < 8]

   # Build dataframe, with column 'label' of int
   import pandas as pd
   import numpy as np
   dataset = pd.DataFrame(X)
   dataset['label'] = pd.Series(y.astype('int'))

   # Split the dataset in training and valitaion sets
   from sklearn.model_selection import train_test_split
   trainset, validset = train_test_split(dataset, test_size = 1/6, random_state = 42)

Before creating the tree, we have to define our custom Pytorch classifier (ref).
Here, I use a very simple feedforward neural network. 

.. code-block:: python

   import torch

   class SimpleClassifier(torch.nn.Module):
       def __init__(self):
           super(SimpleClassifier, self).__init__()
           
           self.linear1 = torch.nn.Linear(784, 16)
           self.linear2 = torch.nn.Linear(16, num_classes)

           self.dropout = torch.nn.Dropout(0.1)
           self.relu = torch.nn.ReLU()

           
       def forward(self, x):
           
           out = self.relu(self.linear1(x))
           out = self.linear2(self.dropout(out))

           return out

   classifier = SimpleClassifier(num_classes=2)

Also, we will need a Pytorch Dataset class (`Pytorch documentation <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_) to handle training.
The __getitem__ function must return the inputs and the label in the form of tuple(dict, label).

.. code-block:: python

   from torch.utils.data import Dataset

   class SimpleDataset(Dataset):
       def __init__(self, dataframe):
           super(SimpleDataset,self).__init__()

           self.inputs = dataframe.drop(['label',], axis=1)
           self.labels = dataframe.label

       def __len__(self):
           return len(self.labels)

       def __getitem__(self, idx):

           sample = torch.tensor(self.inputs.iloc[idx].values, dtype = torch.float)
           label = torch.tensor(self.labels.iloc[idx], dtype = torch.long)

           return {'x':sample}, label

Now create the tree.

.. code-block:: python

   from robertastree import Tree
   tree = Tree(classifier=classifier,
               trainset=trainset,
               validset=validset)

Configure and run the training.

.. code-block:: python

   tree.configure_training(optimizer=torch.optim.SGD,
                           optimizer_params={'lr':2e-3,
                                             'weight_decay':1e-4},
                           loss_function=torch.nn.CrossEntropyLoss(),
                           dataset_class=SimpleDataset,
                           batch_size=256,
                           num_epochs=10,
                           valid_period=100)

   tree.train()

That's it! To use the model for class predictions just run:

.. code-block:: python
   
   tree_output = tree.predict(input, return_probabilities=True)

   # tree_output is a tensor of shape (batchsize, num_classes)
   predicted_class = tree_output.argmax(axis=1)

Regression
----------
Suppose you have to tackle a regression task. 
To each training sample a target in a certain value range (a,b) is assigned.
You can convert the task to a classification task with an arbitrary number of classes N, using RobertasTree dataset utils.

.. code-block:: python
   
   from robertastree.dataset_handling import from_range_to_classes

   dataset["label"], classes = from_range_to_classes(dataset['target'], 
                                                     n_classes=N,
                                                     value_range=(a, b))

Then proceed to training, as described in section `Classification`_.
If you want to go back to a numeric prediction, you can use our inferator:

.. code-block:: python

   from robertastree.inferators import WeightedAverageInferator

   target = WeightedAverageInferator(tree.predict(input), classes)

Visualization
-------------
You can visualize the state of the tree to know the accuracy of each node.
For a simple text visualization run:

.. code-block:: python

   tree.print_status()

For a graphical representation run:

.. code-block:: python

   tree.plot_tree()


Performances
============

We evaluated the performances of the classifier defined in section `Usage`_, both using it on its own and in the tree embedding. 

Here's the best result we got in both cases.

.. csv-table::
   :header: "", "simple model", "tree model"
   :widths: 10, 10, 10

   **accuracy (\%)**, 86.34, 94.02

Significant improvements of the classification accuracy could be obtained by embedding the original classifier in the RobertasTree.
Despite being encouraging, such result is far from being sufficient to establish the usefulness of RobertasTree.
Indeed, we lost the Kaggle competition (forgot to mention?), hence to me it was useless.
The increment of performances in MNIST can be led back to the mere increment of parameters used by the model.
The same improvement could be obtained by adding some hidden units to the original classifier.
Further and systematic tests should be designed, exploring differents tasks and data, seeing if the tree-like structure can get some results unaccessible to the single classifier.

Testing
======

RobertasTree code can be easily tested using pytest testing tool. 
A large list of test can be found `here <https://github.com/lorenzosquadrani/RobertasTree/tree/main/tests>`_. 
You can use the plugin pytest-cov (`documentation <https://pytest-cov.readthedocs.io/en/latest/>`_) to run all the tests and get a coverage report.

.. code-block:: bash

   pip install pytest-cov
   
   cd path/to/RobertasTree
   
   pytest --cov=robertastree tests/


References
==========

- `Pytorch documentation <https://pytorch.org/docs/stable/index.html>`_

- `CommonLitReadibility competition page <https://www.kaggle.com/c/commonlitreadabilityprize>`_



.. |Quality Badge| image:: https://app.codacy.com/project/badge/Grade/54f36e77426e4620b7dd9f8a1b184fbb
   :target: https://www.codacy.com/gh/lorenzosquadrani/RobertasTree/dashboard?utm_source=github.com&utm_medium=referral&utm_content=lorenzosquadrani/RobertasTree&utm_campaign=Badge_Grade

.. |Coverage Badge| image:: https://app.codacy.com/project/badge/Coverage/54f36e77426e4620b7dd9f8a1b184fbb
   :target: https://www.codacy.com/gh/lorenzosquadrani/RobertasTree/dashboard?utm_source=github.com&utm_medium=referral&utm_content=lorenzosquadrani/RobertasTree&utm_campaign=Badge_Coverage)
