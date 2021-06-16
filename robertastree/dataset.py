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
