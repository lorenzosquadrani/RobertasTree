from torch.utils.data import Dataset
from transformers import AutoTokenizer


class RobertasTreeDatasetForTest(Dataset):
    
    def __init__(self, dataset):
        
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        
        self.robertas_data = self.tokenizer(dataset.excerpt.tolist(), 
                            padding = True, 
                            truncation = True,
                            return_tensors = 'pt')
        
        self.ids= dataset.id.tolist()
        
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):       
            
        attention_mask = torch.tensor(self.robertas_data['attention_mask'][idx])
        input_ids =  torch.tensor(self.robertas_data['input_ids'][idx])
        ids = self.ids[idx]
        
        return attention_mask, input_ids, ids



class RobertasTreeDatasetForTrain(Dataset):
    
    def __init__(self, dataset):
        
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        
        self.robertas_data = self.tokenizer(dataset.excerpt.tolist(), 
                            padding = True, 
                            truncation = True,
                            return_tensors = 'pt')
        
        self.labels = dataset.label.tolist()
        
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):       
            
        attention_mask = torch.tensor(self.robertas_data['attention_mask'][idx])
        input_ids =  torch.tensor(self.robertas_data['input_ids'][idx])
        labels = self.labels[idx]
        
        return attention_mask, input_ids,labels





def get_criteria(dataset,classes, i, j):
    

    possible_classes = np.split(classes, 2**(i+1))
    
    first_class, second_class = possible_classes[2*j], possible_classes[2*j+1]
                                
                                
    criteria1 = pd.Series(np.full(len(dataset),False ))
    criteria2 = pd.Series(np.full(len(dataset),False ))
    for label1, label2 in zip(first_class, second_class):
        criteria1 = criteria1 | (dataset["label"] == label1)
        criteria2 = criteria2 | (dataset["label"] == label2)
    
    return criteria1, criteria2





def balance_dataset(dataset):
    counts = list(dataset.label.value_counts())
    labels = list(dataset.label.value_counts().keys())


    mean_count = sum(counts)//len(counts)

    subdata = {}

    for label in dataset.label.unique():
        subdata[label] = dataset[dataset.label == label]

    balanced_dataset = pd.concat(
            [subdata[label].sample(mean_count, replace = True) for label in subdata], axis=0
            )
    
    return balanced_dataset.sample(frac = 1).reset_index()



def from_range_to_classes(targets, n_classes, value_range = None):
    '''
    Divide a numerical range in a given number of intervals, and gives the correspondent labels to a Series
    of numerical values.

    Parameters
    ----------
      targets : pandas.Series
        Targets of the training samples

      n_classes : int
        Number of desired classes

      value_range : float tuple
        The range to divide in classes. If None, the minimum and maximum value from targets are used.

    Returns
    -------
      labels : pandas.Series
        The series containing labels of the training samples in place of the targets

      classes : dict
        The classes labels with the correspondent numerical range (as a float tuple)
    '''

    if value_range is None:
        value_range = (targets.min(), targets.max())
    
    x = np.linspace(value_range[0], value_range[1], n_classes +1 )
    classes = {}
    for i in range(n_classes):
        classes[ str(i) ] = (x[i], x[i+1])

    def get_class(value):
    
        for key in classes:
            if (value>= classes[key][0] and value<=classes[key][1]):
                return int(key)
        raise ValueError("Find a value out of range! I don't feel like going on...")
        
    labels = targets.apply(get_class)
    
    return labels, classes