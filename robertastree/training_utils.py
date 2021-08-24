import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import pandas as pd


def get_optimizer_parameters(model, head_lr=2e-5, embeddings_lr=2e-5,
                             encoder_lr=2e-5, pooler_lr=2e-5, decay_factor=0.95):
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

    encoder_learning_rates = [(encoder_lr * decay_factor**i) for i in range(n_layers)]
    for i in range(n_layers):
        layer_parameters = [x[1] for x in list(model.roberta.named_parameters()) if ('encoder.layer.' + str(i) + '.') in x[0]]
        optimizer_parameters.append({'params': layer_parameters,
                                     'lr': encoder_learning_rates[n_layers - 1 - i]})

    pooler_parameters = [x[1] for x in list(model.roberta.named_parameters()) if 'pooler' in x[0]]
    optimizer_parameters.append({'params': pooler_parameters,
                                 'lr': pooler_lr})

    head_parameters = [x[1] for x in list(model.named_parameters()) if 'roberta' not in x[0]]
    optimizer_parameters.append({'params': head_parameters,
                                 'lr': head_lr})

    return optimizer_parameters


def pretrain_roberta(trainset, batch_size=8, output_path='./', num_epochs=10, mlm_probability=0.15):
    '''
    Pretraining function for roberta-base transformer: perform further pretraining on
    Masked Language Modeling using custom dataset.

    Parameters
    ----------
        trainset : pandas.DataFrame
            The custom dataset. Should contain only a column named "text".

        output_path : str
            The path where the pretrained model will be saved.

        num_epochs : int

        mlm_probability : float
    '''

    if not isinstance(trainset, pd.DataFrame):
        raise TypeError("Error! The dataset should be a pandas dataframe.")

    if not trainset.columns == 'text':
        raise ValueError("Error! The dataset should contain only a column with label \" text \".")

    raw_dataset = Dataset.from_pandas(trainset)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_dataset = raw_dataset.map(tokenize_function,
                                        batched=True,
                                        remove_columns=['text'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_probability)

    trainloader = DataLoader(tokenized_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size)

    model = AutoModelForMaskedLM.from_pretrained('roberta-base')

    optimizer = AdamW(model.parameters(), lr=5e-5)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print("Warning! No cuda device was found. \
               The pretraining will be executed on cpu, but it can take lot of time.")

    model.to(device)
    model.train()

    # Train loop
    for epoch in range(num_epochs):
        train_loss = 0.0

        for batch in trainloader:

            for key in batch:
                batch[key] = batch[key].to(device)

            output = model(**batch)
            loss = output.loss
            loss.backward()

            # Optimizer and scheduler step
            optimizer.step()
            optimizer.zero_grad()

            # Update train loss and global step
            train_loss += loss.item()

        # print summary
        train_loss = train_loss / len(trainloader)
        print('Epoch [{}/{}], Train Loss: {:.4f}'
              .format(epoch + 1, num_epochs, train_loss))

    model.save_pretrained(output_path + '/pretrained')
    print('Pretraining done! The state of roberta tranformer has been saved in the folder \"{}\"'
          .format(output_path + '/pretrained'))
