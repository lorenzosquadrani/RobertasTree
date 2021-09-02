import torch
import numpy as np


def get_probabilities(tree_outputs):
    '''
    Receive the output of a Tree object and return the probabilities associated to each class.

    Parameters
    ----------

        tree_outputs: torch.Tensor

            The result of Tree.predict(input). It is a tensor of shape (batchsize,num_classifiers,2).

    Return
    ------

        probabilities: torch.tensor

            A tensor of shape (batchsize, num_classes), with the probabilities
            that the tree assigned to each class.
    '''

    # tree_outputs has shape (n_classifiers, batchsize, 2)
    # we normalize axis 2 to interpret them as probabilities
    normalized_outputs = torch.nn.functional.softmax(tree_outputs, dim=2)

    # we computed the products of the probabilities to get each class' probability

    nclasses = tree_outputs.shape[1] + 1
    nlayers = int(np.log2(nclasses))

    batch_size = tree_outputs.shape[0]

    probabilities = torch.empty((batch_size, 0)).to(tree_outputs.device)

    for label in range(nclasses):

        p = torch.ones(batch_size).to(tree_outputs.device)

        # this are the decisions (0 or 1, left or right in the tree) which leads to choose the class i
        # note that they are simply the number i expressed in binary form, with nlayers fixed digits
        # ex. nlayers=3; i=0 decisions=[0,0,0], i=1 decisions=[0,0,1], ecc.
        decisions_to_the_class = [int(x) for x in ('{:0' + str(nlayers) + 'b}').format(label)]

        for j in range(nlayers):

            classifier = 0

            for i in range(j):

                classifier += 2**i

            for i, decision in enumerate(reversed(decisions_to_the_class[:j])):

                classifier += 2**i * decision

            # consider class 010 = 2
            # conditions: classifier[j=0]=0, classifier[j=1]=1, classifier[j=2]=4
            # consider class 101 = 5
            # conditions: classifier[j=0]=0, classifier[j=1]=2, classifier[j=2]=5
            # 0 -> 1
            # 1 -> 2
            # 00 -> 3
            # 01 -> 4
            # 10 -> 5
            # 11 -> 6

            p *= normalized_outputs[:, classifier, decisions_to_the_class[j]]

        probabilities = torch.cat([probabilities, p.unsqueeze(dim=1)], dim=1)

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

    batchsize = tree_outputs.shape[0]

    # probabilities shape (batchsize, num_classes)
    probabilities = get_probabilities(tree_outputs)

    print(probabilities.shape)

    # get the mid point of the range of each class
    classes_mean_values = []
    for key in classes:
        classes_mean_values.append((classes[key][0] + classes[key][1]) / 2)

    # regression: weighted average of the midpoints
    target = torch.zeros(batchsize)
    for i in range(len(classes)):
        target += classes_mean_values[i] * probabilities[:, i].squeeze()

    return target
