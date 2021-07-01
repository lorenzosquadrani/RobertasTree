import torch
import numpy as np
from transformers import AutoModel, AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

class Classifier(torch.nn.Module):
    def __init__(self, pretrained_path='roberta-base'):
        super(Classifier, self).__init__()   
        self.roberta = AutoModel.from_pretrained(pretrained_path)
        self.layer_norm = torch.nn.LayerNorm(768)
        self.dropout = torch.nn.Dropout(0.3)
        self.regressor = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        x = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        x = self.regressor(self.dropout(self.layer_norm(x)))
        return x



class Tree:
    
    def __init__(self, classes, inferator, models_path = './', pretrained_path = 'roberta-base'):
        
        self.classes = classes
        self.n_classes = len(classes)
        self.pretrained_path = pretrained_path
        
        self.n_layers = int(np.log2(self.n_classes))
        self.n_outputs = (self.n_classes-1)*2
        
        self.models_path = models_path


        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            print("Warning! No cuda device was found. Operations will be excuted on cpu, very slowly.")

        
        self.classifier = Classifier(pretrained_path=pretrained_path)
        self.classifier = self.classifier.to(self.device)
        
        self.inferator = inferator
    
    def predict(self, input_ids,masks):
        
        self.classifier.eval()

        outputs = torch.empty(0, input_ids.shape[0], 2)
        with torch.no_grad():
            for i in range(self.n_layers):
                n_classifiers = 2**i
                for j in range(n_classifiers):
                    print("DOING CLASSIFIER {}_{}".format(i,j))
                    self.classifier.load_state_dict(self.load_model(i,j))

                    output = self.classifier(input_ids, masks)
                    outputs = torch.cat([outputs, output.unsqueeze(axis=0)])      
        
        return outputs              
    
    
    def load_model(self, i, j):
        return torch.load(self.models_path + 'classifier'+ str(i)+'_'+str(j) + '.pkl',
                                       map_location = self.device)
    
    


    def test_classifier(self, i, j, testloader):
        
        print("Loading classifier {}_{}...".format(i, j), end=' ')
        self.classifier.load_state_dict(self.load_model(i, j))
        self.classifier.eval()
        print("Done!")
        
        possible_classes = self.get_classifier_classes(i, j)

        n_correct = 0
        n_wrong = 0
        n_correct_notin = 0
        n_wrong_notin = 0
        total = 0
        print("Testing...", end=' ')
        with torch.no_grad():
            for mask, input_ids, label in tqdm(testloader):
                mask, input_ids = mask.to(self.device), input_ids.to(self.device)

                output = self.classifier(mask, input_ids)

                for i, choice in enumerate(output.argmax(axis=1).tolist()):
                    chosen_class = possible_classes[choice]
                    other_class = possible_classes[not choice]

                    #più carino ma meno efficiente
                    #n_correct += labels[i] in  chosen_class
                    #n_wrong += labels[i] in other_class
                    #n_correct_notin += labels[i] > max(possible_classes[1]) and choice == 1
                    #n_wrong_notin += labels[i] < min(possible_classes[0]) and choice == 0

                    if label[i] in chosen_class:
                        n_correct += 1
                    elif label[i] in other_class:
                        n_wrong += 1
                    elif label[i] > max(possible_classes[1]):
                        if choice == 1:
                            n_correct_notin += 1
                        elif choice == 0:
                            n_wrong_notin += 1
                    elif label[i] < min(possible_classes[0]):
                        if choice == 0:
                            n_correct_notin += 1
                        elif choice == 1:
                            n_wrong_notin += 1
                    total += 1
        return n_correct, n_wrong, n_correct_notin, n_wrong
        
        
        
    
    def get_classifier_classes(self, i, j):
        possible_classes = np.split(self.classes, 2**(i+1))
        first_class, second_class = tuple(possible_classes[2*j]), tuple(possible_classes[2*j+1])
        return first_class, second_class


    def train_classifier(self, i, j, trainloader, validloader, num_epochs = 5,
                        valid_period=100, output_path = './'):
    

        model = Classifier(pretrained_path=self.pretrained_path)

        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay = 1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0,
                                                    num_training_steps = num_epochs)

        train_loss = 0.0
        best_valid_loss = float('Inf')
        global_step = 0
        
        model.train()
        # Train loop
        for epoch in range(num_epochs):
            train_count = 0
            for batch_data in trainloader:
                
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                labels = batch_data['label'].to(self.device)

                y_pred = model(input_ids=input_ids,  
                               attention_mask=attention_mask)
                
                loss = torch.nn.CrossEntropyLoss()(y_pred, labels)
                loss.backward()
                
                # Optimizer and scheduler step
                optimizer.step()
                optimizer.zero_grad()    
            
                # Update train loss and global step
                train_loss += loss.item()
                global_step += 1
                train_count += 1
                # Validation loop. Save progress and evaluate model performance.
                if global_step % valid_period == 0:
                    model.eval()
                    
                    right = 0
                    total = 0
                    valid_loss = 0.0
                    
                    with torch.no_grad():                    
                        for batch_data in enumerate(validloader):
                            input_ids = batch_data['input_ids'].to(self.device)
                            attention_mask = batch_data['attention_mask'].to(self.device)
                            labels = batch_data['label'].to(self.device)


                            y_pred = model(input_ids=input_ids, 
                                           attention_mask=attention_mask)

                            loss = torch.nn.CrossEntropyLoss()(y_pred, labels)
                            valid_loss += loss.item()
                            
                            total += len(labels)
                            right += (y_pred.argmax(axis = -1) == labels).sum().item()

                    #mean train and validation loss
                    current_train_loss = train_loss / valid_period
                    current_valid_loss = valid_loss / len(validloader)
                    accuracy = right/total*100

                    # print summary
                    print('Epoch: [{}], Step: [{}/{}], train_loss: {:.4f}, valid_loss: {:.4f}, accuracy: {:.2f} \%'
                            .format(epoch, train_count*trainloader.batch_size, 
                                    len(trainloader)*trainloader.batch_size, 
                                    current_train_loss, current_valid_loss, accuracy))
                    
                    # checkpoint
                    if current_valid_loss < best_valid_loss:
                        print("Saved model with best validation loss!")
                        best_valid_loss = current_valid_loss
                        torch.save(model.state_dict(), 
                                   output_path + "classifier{}_{}.bin".format(i, j))

                    train_loss = 0.0         
                    model.train()
            scheduler.step()
        print('Training done!')
