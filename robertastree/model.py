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
        Given a batch of inputs, iteratively load all the classifiers,
        compute their outputs and return them. 
        If return_probabilities is True, the outputs are used to compute
        the probability of each class, and the probabilities are returned.

        Parameters
        ----------------

        inputs : dict

            This variable will be passed to the Pytorch classifier 
            as classifier(**inputs), hence the keys must correspond 
            to the arguments of the forward function (ref).

        Return
        ------

        if return_probabilities is False:

            outputs : torch.tensor

                The outputs of all the classifiers.
                The shape of the tensor is (batch_size,num_classifiers, 2).

        if return_probabilities is True:

            probabilities : torch.tensor

                A tensor of shape (num_classes,) containing the probability
                that the tree has associated to each class.
        '''

        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        outputs = torch.empty(batchsize, 0, 2).to(self.device)

        with torch.no_grad():

            for i in range(self.n_layers):

                n_classifiers = 2**i

                for j in range(n_classifiers):

                    self.classifier.load_state_dict(self.load_model(i, j))

                    output = self.classifier(**inputs)
                    outputs = torch.cat([outputs, output.unsqueeze(dim=1)], dim=1)

        if not return_probabilities:
            return outputs
        else:
            return get_probabilities(outputs)

    def load_model(self, i, j, initial=False):
        '''
        Load and return the state_dict of classifier{i}_{j}.
        If initial=True, the initial state_dict is returned 
        (i and j are ignored).

        Parameters
        --------------

        i,j : int, int

            The indexes of a classifier of the tree.

        Return
        ------

        state_dict: dict

            The state dict of classifier i,j, or the initial state dict if initial=True.

        '''
        if initial:
            return torch.load(self.models_path + 'initial_state', map_location=self.device)
        else:
            return torch.load(self.models_path + 'classifier' + str(i) + '_' + str(j) + '.bin',
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
        '''
        Set the training algorithm and the hyperparameters. This method must be called
        before calling training methods of the tree object.

        Parameters
        ----------

        optimizer : class 

            One the pytorch optimizer classes (SGD, Adam, Momentum, ...). 
            See `pytorch documentation <https://pytorch.org/docs/stable/optim.html>`_ for more details.

        dataset_class : class

            A pytorch custom dataset. 
            The __getitem__ function must return the inputs and the label in the form of tuple(dict, label).
            See `RobertasTree documentation <https://github.com/lorenzosquadrani/RobertasTree#classification>`_
            for an example, and `pytorch documentation <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_
            for further details.

        scheduler : class

            One of the pytorch scheduler classes.
            See `pytorch documentation <https://pytorch.org/docs/stable/optim.html>`_ for more details.

        optimizer_params : dict

            The parameters passed to the optimizer when istantiating it.

        scheduler_params : dict

            The parameters passed to the scheduler when istantiating it.

        dataset_class_params : dict

            The parameters passed to the dataset when istantiating it.

        loss_function : torch.nn.loss

            An istance of a pytorch loss function. 
            See `pytorch documentation <https://pytorch.org/docs/stable/nn.html#loss-functions>`_ for help.

        batch_size : int

        num_epochs : int

        valid_period : int
        '''

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
        self.classifier.load_state_dict(self.load_model(i, j, initial=True))

        # Prepare dataloaders, optimizer, lr scheduler
        trainloader, validloader = self._make_loaders(i, j, self.trainset, self.validset)
        optimizer = self.optimizer(self.classifier.parameters(), **self.optimizer_params)

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer, **self.scheduler_params)

        global_step = 0
        best_valid_loss = float('Inf')

        # Training loop
        self.classifier.train()

        print("[epoch, batch/num_batches]: trainloss | validloss | best_validloss | accuracy")
        for epoch in range(self.num_epochs):

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
                    print('[{:4d}, {:4d}/{:4d}]: {:5.4f} | {:5.4f} | {:5.2f}% | {:5.4f}'
                          .format(epoch, epoch_step,
                                  len(trainloader),
                                  train_loss / epoch_step, valid_loss, accuracy, best_valid_loss))

                    # checkpoint
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(self.classifier.state_dict(),
                                   self.models_path + "classifier{}_{}.bin".format(i, j))
                        self.classifier_accuracy[2**i + j - 1] = accuracy

            if self.scheduler is not None:
                scheduler.step()

    def train(self):

        self._check_is_configured()

        for i in range(self.n_layers):

            n_classifiers = 2**i

            for j in range(n_classifiers):

                print("=" * 10, "Training classifier [{},{}]".format(i, j), "=" * 10)
                self.train_classifier(i, j)
                print('Training of classifier [{},{}] completed!'.format(i, j))

    def test_classifier(self, i, j, testset):

        testloader, _ = self._make_loaders(i, j, testset)

        self.classifier.load_state_dict(self.load_model(i, j))

        _, accuracy = self._validation_step(testloader)

        self.classifier_accuracy[2**i + j - 1] = accuracy

        return accuracy

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
