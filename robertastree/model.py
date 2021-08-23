import torch
import numpy as np
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import robertastree.dataset_handling as dh
from torch.utils.data import DataLoader
import pylab as plt


class Tree:
    '''
    Main class of RobertasTree library. An object Tree is designed to handle
    creation, training, predictions of all the binary classifiers.

    Parameters
    ------------------
    classifier : nn.Module

    trainset : pd.DataFrame

    validset : pd.DataFrame

    models_path : str

    pretrained_path : str
    '''

    def __init__(self, classifier, trainset, validset=None,
                 models_path='./'):

        self.n_classes = len(trainset.label.unique())
        self.n_layers = int(np.log2(self.n_classes))
        self.n_outputs = (self.n_classes - 1) * 2

        self.trainset = trainset
        self.validset = validset

        self.models_path = models_path

        if torch.cuda.is_available():
            self.device = 'cuda'
            print("Found GPU {}. I will use it.".format(torch.cuda.get_device_name(0)))
        else:
            self.device = 'cpu'
            print("Warning! No cuda device was found. Operations will be executed on cpu, very slowly.")

        self.classifier = classifier
        self.classifier = self.classifier.to(self.device)

        torch.save(self.classifier.state_dict(), self.models_path + 'initial_state')

        self.classifier_accuracy = ['?' for i in range(self.n_classes - 1)]

    def predict(self, inputs, batchsize=1):
        '''
        Given a batch of inputs, iteratively load all the classifiers
        and compute their outputs.

        Parameters
        ----------------
        input : dict
        '''

        self.classifier.eval()

        outputs = torch.empty(0, batchsize, 2)
        with torch.no_grad():
            for i in range(self.n_layers):
                n_classifiers = 2**i
                for j in range(n_classifiers):
                    print("DOING CLASSIFIER {}_{}".format(i, j))
                    self.classifier.load_state_dict(self.load_model(i, j))

                    output = self.classifier(**inputs)
                    outputs = torch.cat([outputs, output.unsqueeze(axis=0)])

        return outputs

    def load_model(self, i, j):
        '''
        Load and return the state_dict of classifier{i}_{j}.

        Parameters
        --------------
        i : int

        j : int

        '''
        return torch.load(self.models_path + 'classifier' + str(i) + '_' + str(j) + '.bin',
                          map_location=self.device)

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
                mask, input_ids = mask.to(
                    self.device), input_ids.to(self.device)

                output = self.classifier(mask, input_ids)

                for n, choice in enumerate(output.argmax(axis=1).tolist()):
                    chosen_class = possible_classes[choice]
                    other_class = possible_classes[not choice]

                    # più carino ma meno efficiente
                    # n_correct += labels[i] in  chosen_class
                    # n_wrong += labels[i] in other_class
                    # n_correct_notin += labels[i] > max(possible_classes[1]) and choice == 1
                    # n_wrong_notin += labels[i] < min(possible_classes[0]) and choice == 0

                    if label[n] in chosen_class:
                        n_correct += 1
                    elif label[n] in other_class:
                        n_wrong += 1
                    elif label[n] > max(possible_classes[1]):
                        if choice == 1:
                            n_correct_notin += 1
                        elif choice == 0:
                            n_wrong_notin += 1
                    elif label[n] < min(possible_classes[0]):
                        if choice == 0:
                            n_correct_notin += 1
                        elif choice == 1:
                            n_wrong_notin += 1
                    total += 1
        return n_correct, n_wrong, n_correct_notin, n_wrong

    def _get_classifier_classes(self, i, j):
        classes = np.arange(0, self.n_classes)
        possible_classes = np.split(classes, 2**(i + 1))
        first_class = tuple(possible_classes[2 * j])
        second_class = tuple(possible_classes[2 * j + 1])
        return first_class, second_class

    def train_classifier(self, i, j, batch_size, num_epochs=5,
                         valid_period=1):

        self.classifier.load_state_dict(torch.load(self.models_path + 'initial_state'))

        trainloader = DataLoader(
            dh.RobertasTreeDatasetForClassification(dh.get_subdatasets(self.trainset, i, j)),
            batch_size=batch_size,
            shuffle=True)

        validloader = DataLoader(
            dh.RobertasTreeDatasetForClassification(dh.get_subdatasets(self.validset, i, j)),
            batch_size=batch_size)

        # TODO: handle the initial weights
        # self.classifier.load_state_dict(self.load_model(i, j))

        optimizer = AdamW(self.classifier.parameters(), lr=5e-5, weight_decay=1e-4)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=num_epochs)

        train_loss = 0.0
        best_valid_loss = float('Inf')
        global_step = 0

        self.classifier.train()
        # Training loop
        for epoch in range(num_epochs):
            print("=" * 5, "Starting EPOCH [{}]".format(epoch), "=" * 5)
            train_count = 0
            for batch_data in trainloader:

                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                labels = batch_data['label'].to(self.device)

                y_pred = self.classifier(input_ids=input_ids,
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
                if (global_step % valid_period == 0) and (self.validset is not None):
                    self.classifier.eval()

                    right = 0
                    total = 0
                    valid_loss = 0.0

                    with torch.no_grad():
                        for batch_data in validloader:
                            input_ids = batch_data['input_ids'].to(self.device)
                            attention_mask = batch_data['attention_mask'].to(
                                self.device)
                            labels = batch_data['label'].to(self.device)

                            y_pred = self.classifier(input_ids=input_ids,
                                                     attention_mask=attention_mask)

                            loss = torch.nn.CrossEntropyLoss()(y_pred, labels)
                            valid_loss += loss.item()

                            total += len(labels)
                            right += (y_pred.argmax(axis=-1) == labels).sum().item()

                    # mean train and validation loss
                    current_train_loss = train_loss / valid_period
                    current_valid_loss = valid_loss / len(validloader)
                    accuracy = right / total * 100

                    # print summary
                    print('Step: [{}/{}], train_loss: {:.4f}, valid_loss: {:.4f}, accuracy: {:.2f} %'
                          .format(train_count * trainloader.batch_size,
                                  len(trainloader) * trainloader.batch_size,
                                  current_train_loss, current_valid_loss, accuracy))

                    # checkpoint
                    if current_valid_loss < best_valid_loss:
                        print("Saved model with best validation loss!")
                        best_valid_loss = current_valid_loss
                        torch.save(self.classifier.state_dict(),
                                   self.models_path + "classifier{}_{}.bin".format(i, j))
                        self.classifier_accuracy[2 * i + j] = accuracy

                    train_loss = 0.0
                    self.classifier.train()
            scheduler.step()
        print('Training done!')

    def train(self, batch_size, num_epochs, valid_period):

        for i in range(self.n_layers):
            n_classifiers = 2**i
            for j in range(n_classifiers):
                print("=" * 10, "Training classifier {}_{}".format(i, j), "=" * 10)
                self.train_classifier(i, j,
                                      batch_size=batch_size,
                                      num_epochs=num_epochs,
                                      valid_period=valid_period)

    def plot_tree(self):

        radius = 0.08

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')

        index = 0
        for i in range(self.n_layers):
            n_classifiers = 2**i
            y = (1. / 4) * (3 - i)
            for j in range(n_classifiers):
                x = (1. / (n_classifiers + 1)) * (j + 1)

                accuracy = self.classifier_accuracy[index]
                index += 1
                color = 'red' if accuracy == '?' else 'green'

                circle = plt.Circle((x, y), radius=radius, facecolor=color, alpha=0.5)
                ax.add_patch(circle)

                ax.annotate("{:.3}".format(accuracy), (x, y), fontsize=15, ha='center', va='center')

        fig.show()

        return fig

    def print_status(self):

        index = 0

        for i in range(self.n_layers):
            n_classifiers = 2**i
            for j in range(n_classifiers):
                print("Classifier [{},{}]: accuracy [{:.3}]".format(i, j, self.classifier_accuracy[index]))
                index += 1
