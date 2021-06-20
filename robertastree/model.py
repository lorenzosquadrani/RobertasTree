import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AdamW


class RobertasTreeClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.3, pretrained_path = 'roberta-base'):
        super(RobertasTreeClassifier, self).__init__()
        
        self.roberta = AutoModel.from_pretrained(pretrained_path)
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
    
    def __init__(self, classes, inferator, models_path = './',):
        
        self.classes = classes
        self.n_classes = len(classes)
        
        self.n_layers = int(np.log2(self.n_classes))
        self.n_outputs = (self.n_classes-1)*2
        
        self.models_path = models_path


        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            print("Warning! No cuda device was found. Operations will be excuted on cpu, very slowly.")

        
        self.classifier = RobertasTreeClassifier(0.3)
        self.classifier = self.classifier.to(device)
        
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
    
    
    def load_model(self,i,j):
        return torch.load(self.models_path + 'classifier'+ str(i)+'_'+str(j) + '.pkl',
                                       map_location = self.device)
    
    


    def test_classifier(i,j, testloader):
        
        print("Loading classifier {}_{}...".format(i,j), end = ' ')
        self.classifier.load_state_dict(self.load_model(i,j))
        self.classifier.eval()
        print("Done!")
        
        possible_classes = self.get_classifier_classes(i,j)

        n_correct = 0
        n_wrong  = 0
        n_correct_notin = 0
        n_wrong_notin = 0
        total = 0
        print("Testing...", end = ' ')
        with torch.no_grad():
            for mask,input_ids, label in tqdm(testloader):
                mask, input_ids = mask.to(device), input_ids.to(device)

                output = self.classifier(mask,input_ids)

                for i,choice in enumerate(output.argmax(axis=1).tolist()):
                    chosen_class = possible_classes[choice]
                    other_class = possible_classes[not choice]

                    #più carino ma meno efficiente
                    #n_correct += labels[i] in  chosen_class
                    #n_wrong += labels[i] in other_class
                    #n_correct_notin += labels[i] > max(possible_classes[1]) and choice == 1
                    #n_wrong_notin += labels[i] < min(possible_classes[0]) and choice == 0

                    if labels[i] in chosen_class:
                        n_correct += 1
                    elif labels[i] in other_class:
                        n_wrong += 1
                    elif labels[i] > max(possible_classes[1]):
                        if choice == 1:
                            n_correct_notin += 1
                        elif choice == 0:
                            n_wrong_notin += 1
                    elif labels[i] < min(possible_classes[0]):
                        if choice == 0:
                            n_correct_notin += 1
                        elif choice == 1:
                            n_wrong_notin += 1
                    total += 1
        return n_correct, n_wrong, n_correct_notin, n_wrong
        
        
        
    
    def get_classifier_classes(self,i,j):
        possible_classes = np.split(self.classes, 2**(i+1))
        first_class, second_class = tuple(possible_classes[2*j]), tuple(possible_classes[2*j+1])
        return first_class, second_class


    def train_classifier(self, i,j, trainloader, validloader, num_epochs = 5,
                        valid_period = 100, output_path = './', pretrained_path = 'roberta-base'):
    

        model = RobertasTreeClassifier(0.3, pretrained_path = pretrained_path)

        optimizer = AdamW(get_optimizer_param(self.classifier), lr = 1e-5)
        scheduler = OnFlatStepLR(...)

        train_loss = 0.0

        best_valid_loss = float('Inf')
        
        global_step = 0
        
        model.train()
        
        # Train loop
        for epoch in range(num_epochs):
            for mask,input_ids,labels in trainloader:
                
                mask, input_ids, labels = mask.to(device), input_ids.to(device),labels.to(device)

                y_pred = self.classifier(input_ids=input_ids,  
                               attention_mask=mask)
                
                loss = torch.nn.CrossEntropyLoss()(y_pred, labels)
                
                loss.backward()
                
                
                # Optimizer and scheduler step
                optimizer.step()    
                    
                optimizer.zero_grad()
                
                # Update train loss and global step
                train_loss += loss.item()
                global_step += 1

                # Validation loop. Save progress and evaluate model performance.
                if global_step % valid_period == 0:
                    model.eval()
                    
                    right = 0
                    total = 0
                    valid_loss = 0.0
                    
                    with torch.no_grad():                    
                        for mask, input_ids, labels in validloader:
                            mask, input_ids, labels = mask.to(device), input_ids.to(device),labels.to(device)

                            y_pred = model(input_ids=input_ids, 
                                           attention_mask=mask)

                            loss = torch.nn.CrossEntropyLoss()(y_pred, labels)
                            valid_loss += loss.item()
                            
                            total += len(labels)
                            right += (y_pred.argmax(axis = -1) == labels).sum().item()

                    # Store train and validation loss history
                    train_loss = train_loss / valid_period
                    valid_loss = valid_loss / len(validloader)
                    accuracy = right/total*100

                    # print summary
                    print('Epoch [{}/{}], global step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Accurracy: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, num_epochs*len(trainloader),
                                  train_loss, valid_loss, accuracy))
                    
                    # checkpoint
                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(self.classifier.state_dict(), '/classifier' + str(i)+ '_'  +str(j) + '.pkl')
                        save_checkpoint(output_path + '/classifier' + str(i)+ '_'  +str(j) + '.pkl', model, best_valid_loss)
                        print("Saved the model state dict with best validation!")
                    
                            
                    train_loss = 0.0                
                    model.train()
            scheduler.step(valid_loss, self.classifier)
        print('Training done!')