from torch.utils.data import Dataset



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
