def RobertasTreeInferator(tree_outputs, classes):
    '''
    Receives the output of a RobertasTree model (a tensor of shape (n_classes-1, 2))
    
    Return the a number between -4 and 2.
    '''
    
    nclasses = len(classes)
    nlayers  = int(np.log2(nclasses))
    
    normalized_outputs = torch.nn.functional.softmax(tree_outputs, dim = 1)
    # =========== Method 1 - follow the tree ===========
    
    
    # =========== Method 2 - compute probabilities and weighted sum ============
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
    
    # =========== Method 3 - compute probabilities and weighted sum with squared ==========
    
    