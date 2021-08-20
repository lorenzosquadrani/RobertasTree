| **Authors**  | **Project** |  **Documentation** | **Build Status** | **Code Quality** | **Coverage** |
|:------------:|:-----------:|:------------------:|:----------------:|:----------------:|:------------:|
| [**L. Squadrani**](https://github.com/lorenzosquadrani) <br/> | **RobertasTree** <br/> | **Docs:**  |  **Linux:** <br/> | **Codacy:** <br/> <br/> **Codebeat:** <br/>  | **CodeCov:** <br/>  |


<a href="https://www.google.com/search?q=pera&oq=pera&aqs=chrome..69i57.783j0j1&sourceid=chrome&ie=UTF-8">
  <div class="image">
    <img src="https://icons.iconarchive.com/icons/alex-t/fresh-fruit/256/pear-icon.png" width="90" height="90">
  </div>
</a>


# RobertasTree

## A Tree-like model for Multiclass Natural Language Classification based on Roberta

Implementation and optimization of Roberta-based Deep Learning model developed for Kaggle competition [CommonLitReadibility](https://www.kaggle.com/c/commonlitreadabilityprize).

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
* [Testing](#testing)
* [References](#references)



## Overview
The idea behind RobertasTree model is very simple and general. Consider a N-classes classification task, where N is a power of 2. This task can be decomposed in N-1 binary classification tasks, organized in a tree-like structure like in figure. Such approach could be advantageous or disadvantageous according to your data. In general, the advantages that you may get are the following (not verified, just guessing):

* as the same data are utilized to train different classifier on different binary class, a certain form of **data augmentation** is obtained.
* the number of parameters of the model are increased, without increasing overfitting risk (I hope).

Unfortunately, for now my code is not as general as the idea. I developed a tree-like model for text classification. Classifiers are expected to be composed by Roberta's transformer and a head. Training is implemented for the fine-tuning of this kind of model.


## Prerequisites
* numpy
* pandas
* transformers
* torch


## Installation
The installation steps are

```bash
python -m pip install -r ./requirements.txt
```

to install the prerequisites and then

```bash
python setup.py install
```

or for installing in development mode:

```bash
python setup.py develop --user
```

## Usage
Here there is the pipeline to apply a RobertasTree model on CommonLitReadibilityPrize regression task (I do not recommend to try it on your easy-running-out-of-memory laptop. Use Google Colab instead.)

First we prepare the dataset. We load the competitition csv, convert the targets into class labels using the apposite function in from_range_to_classes.

```python
import pandas as pd
from robertastree.dataset_handling import from_range_to_classes

dataset = pd.read_csv('/content/train.csv')
dataset["label"], classes = from_range_to_classes(dataset['target'], 
                                                  n_classes=8,
                                                  value_range=(-4., 2.))
```
Eventually, you can split the dataset in a training set and a validation set.
```python
from sklearn.model_selection import train_test_split

trainset, validset = train_test_split(dataset, test_size = 0.2, random_state = 42)
```

Before creating the tree, you have to define your Pytorch classifier. Here, I use the one me and my team designed for the CommonLit competition.
```python
import torch
from transformers import AutoModel

class CommonLitClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CommonLitClassifier, self).__init__()
        
        self.roberta = AutoModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, 2)
        
    def forward(self, input_ids, attention_mask):
        x = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x = self.d1(x)
        x = (self.l1(x))
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)
        
        return x

classifier = CommonLitClassifier()
```

Now create the tree.
```python
from robertastree import Tree
tree = Tree(classifier=classifier,
            trainset=trainset,
            validset=validset)
```

## Testing
TO DO

## References
TO DO
