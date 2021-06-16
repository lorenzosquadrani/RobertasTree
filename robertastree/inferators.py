def RobertasTreeInferator(tree_outputs, classes):
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
    '''
    nclasses = len(classes)
    nlayers  = int(np.log2(nclasses))
    
    normalized_outputs = torch.nn.functional.softmax(tree_outputs, dim = 1)

    layer1, layer2, layer3 = torch.split(normalized_outputs, [2**i for i in range(nlayers)], dim = 0)
    
    probabilities = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                probabilities[str(4*i + 2*j + k)] = (layer1[0,i]*layer2[i,j]*layer3[2*i + j,k]).item()
    
    classes_mean_value = {}
    for key in classes:
        classes_mean_value[key] = (classes[key][0]+ classes[key][1])/2
        
    p = 1
    for key in classes:
        print(probabilities[key])
        p *=   classes_mean_value[key] * (probabilities[key])

    return p
    
    