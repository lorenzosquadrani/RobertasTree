import torch
import numpy as np


def get_probabilities(tree_outputs):

    # tree_outputs has shape (n_classifiers, batchsize, 2)
    # we normalize axis 2 to interpret them as probabilities
    normalized_outputs = torch.nn.functional.softmax(tree_outputs, dim=2)

    # we computed the products of the probabilities to get each class' probability

    nclasses = tree_outputs.shape[0] + 1
    nlayers = int(np.log2(nclasses))

    probabilities = {}

    for i in range(nclasses):

        p = 1.

        # this are the decisions (0 or 1, left or right in the tree) which leads to choose the class i
        # note that they are simply the number i expressed in binary form, with nlayers fixed digits
        # ex. nlayers=3; i=0 decisions=[0,0,0], i=1 decisions=[0,0,1], ecc.
        decisions_to_the_class = [int(x) for x in ('{:0' + str(nlayers) + 'b}').format(i)]

        for j in range(nlayers):
            p *= normalized_outputs[j, :, decisions_to_the_class[j]].item()

        probabilities[i] = p

    return probabilities


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

    probabilities = get_probabilities(tree_outputs)

    # get the mid point of the range of each class
    classes_mean_value = {}
    for key in classes:
        classes_mean_value[key] = (classes[key][0] + classes[key][1]) / 2

    # regression: weighted average of the midpoints
    p = 0.
    for key in classes:
        p += classes_mean_value[key] * (probabilities[key])

    return p
