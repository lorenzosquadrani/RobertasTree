
def get_optimizer_parameters(model, head_lr = 2e-5, embeddings_lr = 2e-5,
                             encoder_lr = 2e-5 , pooler_lr = 2e-5, decay_factor = 0.95):

    '''
    Get learning rates for the optimizer, allowing layer-wise differentiation

    Parameters
    ----------
        model : RobertasTree
            the model to be trained

        head_lr : float

        embedding_lr : float

        encoder_lr : float

        pooler_lr : float

        decay_factor : float
            the learning rate of roberta's encoder decays from top to bottom with this factor

    Returns
    -------
      self
    '''
    optimizer_parameters = []
    n_layers = 12
    
    
    embeddings_parameters = [x[1] for x in list(model.roberta.named_parameters()) if 'embeddings' in x[0]]
    optimizer_parameters.append({'params': embeddings_parameters,
                                 'lr': embeddings_lr})
    
    encoder_learning_rates = [(encoder_lr * decay_factor**i)  for i in range(n_layers)]
    for i in range(n_layers):
        layer_parameters = [x[1] for x in list(model.roberta.named_parameters()) if ('encoder.layer.'+str(i)+'.') in x[0]] 
        optimizer_parameters.append({'params': layer_parameters,
                                   'lr': encoder_learning_rates[n_layers - 1 -i]})
        
    
    pooler_parameters = [x[1] for x in list(model.roberta.named_parameters()) if 'pooler' in x[0]]
    optimizer_parameters.append({'params': pooler_parameters,
                                 'lr': pooler_lr})
    
    head_parameters = [x[1] for x in list(model.named_parameters()) if 'roberta' not in x[0]]
    optimizer_parameters.append({'params': head_parameters,
                                 'lr': head_lr})
        
    return optimizer_parameters



class OnFlatStepLR:
    '''
    Stochastic Gradient Descent with Momentum specialiation
    Update the parameters according to the rule

    .. code-block:: python

        v = momentum * v - learning_rate * gradient
        parameter += v - learning_rate * gradient


    Parameters
    ----------
        optimizer : torch.optim.optimizer

        n_steps : int

        epochs_to_wait : int

        gamma : float

        restart_from_best : bool

        best_path : str


  '''
    
    def __init__(self, optimizer, n_steps, epochs_to_wait , gamma = 0.1,
                 restart_from_best = False, best_path = None):
        self.optimizer = optimizer
        self.n_steps = n_steps
        self.epochs_to_wait = epochs_to_wait
        self.gamma = gamma
        self.restart_from_best = restart_from_best
        if restart_from_best:
            self.best_path = best_path
            
        self.best_loss = float('Inf')
        self.epochs_without_improving = 0
        self.steps_done = 0
            
    
    def step(self, current_loss, model): 
        
        if self.steps_done>= self.n_steps:
            return None
        
        if current_loss<self.best_loss:
            self.best_loss = current_loss
            self.epochs_without_improving = 0
            return None
        else:
            self.epochs_without_improving += 1
            
        if self.epochs_without_improving >= self.epochs_to_wait:
            
            #Load the best previous state, before starting with new lower learning rate
            if self.restart_from_best:
                state_dict = torch.load(self.best_path, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                print('-'*30)
                print("Changed learning rate and loaded model with validation loss {:.4f}!"
                      .format(state_dict["valid_loss"]))
                
            for x in optimizer.param_groups:
                x['lr'] *= self.gamma
                
            self.epochs_without_improving = 0
            self.steps_done += 1  