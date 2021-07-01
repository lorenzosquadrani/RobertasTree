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
TO DO

## Testing
TO DO

## References
TO DO
