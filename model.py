import torch




class RobertasTreeClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(RobertasTreeClassifier, self).__init__()
        
        self.roberta = AutoModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, 2)
        
    def forward(self, input_ids, attention_mask):
        x = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x = self.d1(x)
        x = (self.l1(x))
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)
        
        return x



class RobertasTree:
    
    def __init__(self, classes, inferator, models_path = './', device = 'cuda'):
        
        self.classes = classes
        self.n_classes = len(classes)
        
        self.n_layers = int(np.log2(self.n_classes))
        self.n_outputs = (self.n_classes-1)*2
        
        self.models_path = models_path
        self.device = device
        
        self.classifier = RobertasTreeClassifier(0.3)
        self.classifier = self.classifier.to(device)
        
        self.inferator = inferator
    
    def predict(self, input_ids,masks):
        
        outputs = torch.empty(0, input_ids.shape[0], 2)
        for i in range(self.n_layers):
            n_classifiers = 2**i
            for j in range(n_classifiers):
                print("DOING CLASSIFIER {}_{}".format(i,j))
                self.classifier.load_state_dict(self.load_model(i,j))

                output = self.classifier(input_ids, masks)
                outputs = torch.cat([outputs, output.unsqueeze(axis=0)])      
        
        return outputs              
    
    
    def load_model(self,i,j):
        return torch.load(self.models_path + 'best_model_classifier'+ str(i)+'_'+str(j) + '.pkl',
                                       map_location = self.device)['model_state_dict']
    
    
    def test_classifier(i,j, testloader):
        
        print("Loading classifier...", end = ' ')
        self.classifier.load_state_dict(self.load_model(i,j))
        print("Done!")
        
        predictions, labels = [],[]
        possible_classes = self.get_classifier_classes(i,j)
        
        print("Making predictions...", end = ' ')
        with torch.no_grad():
            for mask,input_ids, label in tqdm(testloader):
                mask, input_ids = mask.to(device), input_ids.to(device)

                output = self.classifier(mask,input_ids)
                predictions.extend(self.get_labels(i,j, classifier_output))
                labels.extend(label)
        print("Done!")
        
        n_correct = 0
        n_wrong  = 0
        n_correct_notin = 0
        n_wrong_notin = 0
        
        for i in range(len(labels)):
            if labels[i] in predictions[i]:
                n_correct += 1
            elif labels[i] in possible_cla
        
        
        
        
        
    def get_labels(self,i,j, classifier_output):
        
        choices = classifier_output.argmax(dim = 1)
    
        classes = self.get_classifier_classes(i,j)
        
        labels = []
        for choice in choices:
            labels.append(classes[int(choice)])
                
        return labels
    
    
    def get_classifier_classes(self,i,j):
        possible_classes = np.split(self.classes, 2**(i+1))
        first_class, second_class = tuple(possible_classes[2*j]), tuple(possible_classes[2*j+1])
        return first_class, second_class