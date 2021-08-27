import torch
import numpy as np


def get_probabilities(tree_outputs):
    '''
    Receive the output of a Tree object and return the probabilities associated to each class.

    Parameters
    ----------

        tree_outputs: torch.Tensor
            The result of Tree.predict(input). It is a tensor of shape (nclassifier,batchsize,2).

    Return
    ------

        probabilities: dict
            A dict in which the keys are the classes' labels (0,1,2...,nclasses) and the values are
            the probabilities the tree assigned to each class. 
            Each value is a tensor of shape (batchsize,)
    '''

    # tree_outputs has shape (n_classifiers, batchsize, 2)
    # we normalize axis 2 to interpret them as probabilities
    normalized_outputs = torch.nn.functional.softmax(tree_outputs, dim=2)

    # we computed the products of the probabilities to get each class' probability

    nclasses = tree_outputs.shape[0] + 1
    nlayers = int(np.log2(nclasses))

    probabilities = {}

    for i in range(nclasses):

        p = torch.ones(tree_outputs.shape[1])

        # this are the decisions (0 or 1, left or right in the tree) which leads to choose the class i
        # note that they are simply the number i expressed in binary form, with nlayers fixed digits
        # ex. nlayers=3; i=0 decisions=[0,0,0], i=1 decisions=[0,0,1], ecc.
        decisions_to_the_class = [int(x) for x in ('{:0' + str(nlayers) + 'b}').format(i)]

        for j in range(nlayers):
            p *= normalized_outputs[j, :, decisions_to_the_class[j]]

        probabilities[i] = p

    return probabilities


def WeightedAverageInferator(tree_outputs, classes):
    ''' 
    Elaborate the outputs of a RobertaTree model to infere the best value.

    Parameters
    ----------
      tree_outputs : torch.tensor
        Output of the Roberta Tree model. It is a tensor of shape (nclasses, batchsize,2)

      classes : dict
        Dictionary with classes' name as keys and tuple with class numerical interval as values

    Returns
    -------
      targets: torch.tensor
          The results of the regression, given by an average of the classes values,
          weighted with the probabilities predicted by RobertasTree. 
          It has shape (batchsize,).
    '''

    probabilities = get_probabilities(tree_outputs)

    # get the mid point of the range of each class
    classes_mean_value = {}
    for key in classes:
        classes_mean_value[key] = (classes[key][0] + classes[key][1]) / 2

    # regression: weighted average of the midpoints
    target = torch.zeros(tree_outputs.shape[1])
    for key in classes:
        target += classes_mean_value[key] * (probabilities[key])

    return target
