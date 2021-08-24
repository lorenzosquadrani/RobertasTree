import torch
import numpy as np


def WeightedAverageInferator(tree_outputs, classes):
    ''' 
    Elaborate the outputs of a RobertaTree model to infere the best value.

    Parameters
    ----------
      tree_outputs : torch.tensor
        Output of the Roberta Tree model

      classes : dict
        Dictionary with classes' name as keys and tuple with class numerical interval as values

    Returns
    -------
      float

      The results of the regression, given by an average of the classes values,
      weighted with the probabilities predicted by RobertasTree.
    '''

    nclasses = len(classes)
    nlayers = int(np.log2(nclasses))

    # apply softmax for probabilistic interpretation
    normalized_outputs = torch.nn.functional.softmax(tree_outputs, dim=1)

    # split outputs in one array for each tree's layer
    layer1, layer2, layer3 = torch.split(normalized_outputs, [2**i for i in range(nlayers)], dim=0)

    # comput the 2^{nlayers}=nclasses probabilities, each probability is the product of nlayers factors
    probabilities = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                probabilities[4 * i + 2 * j + k] = (layer1[0, i] * layer2[i, j] * layer3[2 * i + j, k]).item()

    # get the mid point of the range of each class
    classes_mean_value = {}
    for key in classes:
        classes_mean_value[key] = (classes[key][0] + classes[key][1]) / 2

    # regression: weighted average of the midpoints
    p = 0.
    for key in classes:
        p += classes_mean_value[key] * (probabilities[key])

    return p
