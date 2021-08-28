import torch
import numpy as np
import pylab as plt

import robertastree.dataset_handling as dh
from robertastree.inferators import get_probabilities


class Tree:

    '''
    Main class of RobertasTree library. An object Tree is designed to handle
    creation, training, predictions of all the binary classifiers.

    Parameters
    ------------------
    classifier : nn.Module
        A custom Pytorch classifier (ref). 
        If a cuda device is available, the classifier will be stored on GPU.

    trainset : pd.DataFrame
        The dataframe must contain a column of int named "label", which will be 
        used as samples' labels.

    validset : pd.DataFrame
        The dataframe must contain a column of int named "label", which will be 
        used as samples' labels.

    models_path : str
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

        self.configured = False

    def _check_is_configured(self):
        if not self.configured:
            raise RuntimeError('The tree has not been configured yet.\nYou should call the function'
                               'Tree.configure before training or testing the classifiers. See ref'
                               ' for help.')

    def predict(self, inputs, batchsize=1, return_probabilities=False):
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

        if not return_probabilities:
            return outputs
        else:
            return get_probabilities(outputs)

    def load_model(self, i, j, initial=False):
        '''
        Load and return the state_dict of classifier{i}_{j}.

        Parameters
        --------------
        i : int

        j : int

        '''
        if initial:
            torch.load(self.models_path + 'initial_state', map_location=self.device)
        else:
            torch.load(self.models_path + 'classifier' + str(i) + '_' + str(j) + '.bin',
                       map_location=self.device)

    def _get_classifier_classes(self, i, j):

        classes = np.arange(0, self.n_classes)

        # the classifier have to distinguish between macro-classes composed
        # by num_classes/2**(i+1) single classes
        possible_classes = np.split(classes, 2**(i + 1))

        first_class = tuple(possible_classes[2 * j])
        second_class = tuple(possible_classes[2 * j + 1])

        return first_class, second_class

    def _make_loaders(self, i, j, trainset, validset=None):

        trainloader, validloader = None, None

        trainloader = torch.utils.data.DataLoader(
            self.dataset_class(dh.get_subdatasets(self.trainset, i, j)),
            batch_size=self.batch_size,
            shuffle=True)

        if validset is not None:

            validloader = torch.utils.data.DataLoader(
                self.dataset_class(dh.get_subdatasets(self.validset, i, j)),
                batch_size=self.batch_size)

        return trainloader, validloader

    def configure(self, optimizer,
                  dataset_class,
                  scheduler=None,
                  optimizer_params=None,
                  scheduler_params=None,
                  dataset_class_params=None,
                  loss_function=torch.nn.CrossEntropyLoss(),
                  batch_size=1, num_epochs=1, valid_period=1):

        self.optimizer = optimizer
        self.optimizer_params = optimizer_params if optimizer_params is not None else {}
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params if scheduler_params is not None else {}
        self.loss_function = loss_function
        self.dataset_class = dataset_class
        self.dataset_class_params = dataset_class_params if dataset_class_params is not None else {}
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.valid_period = valid_period

        self.configured = True

    def _validation_step(self, validloader):
        self.classifier.eval()

        right = 0
        total = 0
        loss = 0.

        with torch.no_grad():

            for samples, labels in validloader:

                for key in samples:
                    samples[key] = samples[key].to(self.device)

                labels = labels.to(self.device)

                y_pred = self.classifier(**samples)

                loss_ = self.loss_function(y_pred, labels)
                loss += loss_.item()

                total += len(labels)
                right += (y_pred.argmax(axis=-1) == labels).sum().item()

                valid_loss = loss / len(validloader)
                accuracy = right / total * 100

        self.classifier.train()

        return valid_loss, accuracy

    def train_classifier(self, i, j):

        # Check training algirithm has been configured
        self._check_is_configured()

        # Reset the classifier to the initial state
        self.load_model(i, j, initial=True)

        # Prepare dataloaders, optimizer, lr scheduler
        trainloader, validloader = self._make_loaders(i, j, self.trainset, self.validset)
        optimizer = self.optimizer(self.classifier.parameters(), **self.optimizer_params)

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.scheduler_params)

        global_step = 0
        best_valid_loss = float('Inf')

        # Training loop
        self.classifier.train()
        for epoch in range(self.num_epochs):

            print("Starting EPOCH [{}]".format(epoch))

            train_loss = 0.
            epoch_step = 0

            for samples, labels in trainloader:

                for key in samples:
                    samples[key] = samples[key].to(self.device)

                labels = labels.to(self.device)

                y_pred = self.classifier(**samples)

                loss = self.loss_function(y_pred, labels)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                global_step += 1
                epoch_step += 1

                # Validation loop. Save progress and evaluate model performance.
                if (global_step % self.valid_period == 0) and (validloader is not None):

                    valid_loss, accuracy = self._validation_step(validloader)

                    # print summary
                    print('Step: [{}/{}], train_loss: {:.4f}, valid_loss: {:.4f}, accuracy: {:.2f} %'
                          .format(epoch_step * self.batch_size,
                                  len(trainloader) * self.batch_size,
                                  train_loss / epoch_step, valid_loss, accuracy))

                    # checkpoint
                    if valid_loss < best_valid_loss:
                        print("Saved model with best validation loss!")
                        best_valid_loss = valid_loss
                        torch.save(self.classifier.state_dict(),
                                   self.models_path + "classifier{}_{}.bin".format(i, j))
                        self.classifier_accuracy[2**i + j - 1] = accuracy

            print("train loss: {:.4f}, best validation loss: {:.4f}".format(train_loss / epoch_step,
                                                                            best_valid_loss))

            if self.scheduler is not None:
                scheduler.step()

    def train(self):

        for i in range(self.n_layers):
            n_classifiers = 2**i
            for j in range(n_classifiers):
                print("=" * 10, "Training classifier [{},{}]".format(i, j), "=" * 10)
                self.train_classifier(i, j)
                print('Training of classifier [{},{}] completed!'.format(i, j))

    def test_classifier(self, i, j, testset):

        self._check_is_configured()

        testloader, _ = self._make_loaders(i, j, testset)

        print("Loading classifier {}_{}...".format(i, j), end=' ')
        self.load_model(i, j)
        print("Done!")

        possible_classes = self._get_classifier_classes(i, j)

        right = 0
        wrong = 0
        right_notin = 0
        wrong_notin = 0
        total = 0

        print("Testing...", end=' ')

        with torch.no_grad():

            for samples, labels in testloader:

                for key in samples:
                    samples[key] = samples[key].to(self.device)

                labels = labels.to(self.device)

                y_pred = self.classifier(**samples)

                for n, choice in enumerate(y_pred.argmax(axis=1).tolist()):
                    chosen_class = possible_classes[choice]
                    other_class = possible_classes[not choice]

                    right += labels[i] in chosen_class
                    wrong += labels[i] in other_class
                    right_notin += labels[i] > max(possible_classes[1]) and choice == 1
                    wrong_notin += labels[i] < min(possible_classes[0]) and choice == 0

                    total += 1

        return right, wrong, right_notin, wrong

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

        return fig

    def print_status(self):

        index = 0

        for i in range(self.n_layers):
            n_classifiers = 2**i
            for j in range(n_classifiers):
                print("Classifier [{},{}]: accuracy [{:.3}]".format(i, j, self.classifier_accuracy[index]))
                index += 1
