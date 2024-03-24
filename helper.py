import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


###########################################################################

# File includes functions related to training and basic performance evaluation

###########################################################################


def train_model_default(model, data_loader, loss_function, optimizer, scheduler, epochs, legal_bert=False):
    """
    Default training process (WITHOUT variable learning rate, remapping targets, and custom class weight loss)
    (explicit training function, called in larger train_test function)
    - grabs pre-trained sentence embeddings stored locally
    - trains model document by document across all 246 documents
    - averages final loss across all documents

    Parameters:
    - model : type of model used to train the data (BiLSTM or CNN_BiLSTM)
    - data_loader : function to gather embeddings
    - loss_function
    - optimizer
    - epochs : number of epochs you wish to train on


    Returns:
    - average loss across all documents (numpy float64 )
    """
    model.train()
    losses_over_epochs = []
    running_lr = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        #iterate over all documents
        for doc_idx in tqdm(range(246)): 
            if legal_bert:
                TRAIN_emb = data_loader(filepath=f"train_document/doc_{doc_idx}_legal/embedding")
                TRAIN_labels = data_loader(filepath=f"train_document/doc_{doc_idx}_legal/label")
            else:
                TRAIN_emb = data_loader(filepath=f"train_document/doc_{doc_idx}/embedding")
                TRAIN_labels = data_loader(filepath=f"train_document/doc_{doc_idx}/label")

                if TRAIN_emb.size(0) == 0: #ignore any faulty embeddings
                    continue
                output = model(TRAIN_emb) #push embeddings through model 
                loss = loss_function(output, TRAIN_labels) #calculate loss for document
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        print(f"Epoch: {epoch+1} | Document: {doc_idx+1}/246 | Loss: {loss.item():.5f}")
        running_lr.append(optimizer.state_dict()['param_groups'][0]['lr'])      #keep track of the variable learning rates over epochs

        losses_over_epochs.append(loss.item()) #store final loss for each document, then average across all documents
    # return np.mean(losses_over_epochs)
    return np.mean(losses_over_epochs), running_lr, losses_over_epochs, model



def test_model(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score, legal_bert=False):
    """
    (explicit evaluation function, called in larger train_test function)
    - grabs test embeddings and labels for each test document
    - evaluates trained model and returns evaluation metrics

    Parameters:
    - model (class): type of model used to train the data (BiLSTM or CNN_BiLSTM)
    - data_loader (function): function to gather embeddings
    - calculate_confusion_matrix (function): function to build a confusion matrix with the given results
    - class_accuracy (function): function to calculate the accuracy with respect to a single class
    - class_f1_score (function):  function to calculate the f1-score with respect to a single class

    Returns:
    - accuracies (numpy array): accuracies for each individual class
    - f1_scores (numpy array): f1-scores for each individual class 
    - average_accuracy (numpy float64): average accuracy across all classes
    - average_f1 (numpy float64): average f1-score across all classes
    """
    confusion_matrix = None
    for i in range(29):
        if legal_bert:
            TEST_emb = data_loader(filepath=f"test_document/doc_{i}_legal/embedding")
            TEST_labels = data_loader(filepath=f"test_document/doc_{i}_legal/label")
        else:
            TEST_emb = data_loader(filepath=f"test_document/doc_{i}/embedding")
            TEST_labels = data_loader(filepath=f"test_document/doc_{i}/label")

        conf_matrix_helper = calculate_confusion_matrix(TEST_emb, TEST_labels, model)
        if confusion_matrix is None: #if no confusion matrix is given, use default confusion matrix
            confusion_matrix = conf_matrix_helper 
        else:
            confusion_matrix = np.add(confusion_matrix, conf_matrix_helper)
            
    accuracies = class_accuracy(confusion_matrix) #gather accuracies for each class from confusion matrix
    f1_scores = class_f1_score(confusion_matrix) #gather f1_scores for each class from confusion matrix
    average_accuracy = np.mean(accuracies)
    average_f1 = np.mean(f1_scores)

    return accuracies, f1_scores, average_accuracy, average_f1, confusion_matrix


def default_train_test(parameters, model, legal_model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score):
    """
    (Overhead function called in main.py to train and evaluate model)

    - train and test a single model given the parameters outlined in 'main.py'

    Parameters:
    - parameters (dictionary): hyperparameters given to the model (layers, hidden size, epochs etc.)
    - model (class): type of model used to train the data (BiLSTM or CNN_BiLSTM)
    - data_loader (function): function to gather embeddings
    - calculate_confusion_matrix (function): function to build a confusion matrix with the given results
    - class_accuracy (function): function to calculate the accuracy with respect to a single class
    - class_f1_score (function):  function to calculate the f1-score with respect to a single class

    Returns:
    - result (list) : object display both the parameters used to train the model and evaluation metrics on how the model performed
    """
    result = []             #instantiate list object to hold info on parameters/evaluation

    #define model optimizer and loss function
    model_opt = torch.optim.Adam(model.parameters(), lr= parameters['learning_rate']) #define model optimizer and loss function
    legal_model_opt = torch.optim.Adam(legal_model.parameters(), lr= parameters['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_opt, T_max= parameters['epochs'],
                                                           eta_min= parameters['learning_rate_floor'])
    loss_function = nn.CrossEntropyLoss()
    print("\nWorking with: ")
    print(parameters)
    print(f"Model type: {model.__class__.__name__}")
    print("Train type: default")
    print("\n")


    print(f'{"Starting Training":-^100}')
    avg_loss, running_lr, loss_over_epochs, model= train_model_default(model, data_loader, loss_function, model_opt, scheduler, parameters['epochs']) #train model

    #test model
    accuracies, f1_scores, average_accuracy, average_f1, confusion_matrix = test_model(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score)
    print(f'accuracies {type(accuracies)}')

    print("Accuracies: {} \n Average accuracy: {}".format(accuracies, average_accuracy))
    print("F1 Scores: {} \n Average F1: {}".format(f1_scores, average_f1))
    
    result.append((parameters, (average_accuracy, average_f1)))


    # return result
    return result, confusion_matrix, running_lr, loss_over_epochs, model




def grid_search_train_test(parameters, model, legal_model, grid_search, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score):
    """
    (Overhead function called in main.py to train and evaluate model)

    - train and test multiple models using a grid search technique to test all possible input hyperparameters defined in 'main.py'

    Parameters:
    - parameters (dictionary): hyperparameters given to the model (layers, hidden size, epochs etc.)
    - model (class): type of model used to train the data (BiLSTM or CNN_BiLSTM)
    - data_loader (function): function to gather embeddings
    - calculate_confusion_matrix (function): function to build a confusion matrix with the given results
    - class_accuracy (function): function to calculate the accuracy with respect to a single class
    - class_f1_score (function):  function to calculate the f1-score with respect to a single class

    Returns:
    - result (list) : object display both the parameters used to train the model and evaluation metrics on how the model performed

    """
    result = []             #instantiate list object to hold info on parameters/evaluation

    parameter_configs = grid_search(parameters)         #gather parameters for the grid search defined in main.py
    
    #iterate over all possible hyperparameter configurations 
    for config in parameter_configs:

        model_opt = torch.optim.Adam(model.parameters(), lr= config['learning_rate'])       #define model optimizer and loss function
        legal_model_opt = torch.optim.Adam(legal_model.parameters(), lr= config['learning_rate'])
        loss_function = nn.CrossEntropyLoss()
        print("\nWorking with: ")
        print(config)
        print(f"Model type: {model.__class__.__name__}")
        print("Train type: grid search")

        print("\n")
        
        print(f'{"Starting Training":-^100}')
        epochs = config['epochs']
        
        avg_loss, running_lr, loss_over_epochs, model = train_model_default(model, data_loader, loss_function, model_opt, epochs)
    
        avg_loss_legal, running_lr_legal, loss_over_epochs_legal, legal_model = train_model_default(legal_model, data_loader, loss_function, legal_model_opt, epochs, True)
    
        accuracies_base, f1_scores_base, average_accuracy_base, average_f1_base, confusion_matrix = test_model(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score)
    
        accuracies_base_legal, f1_scores_base_legal, average_accuracy_base_legal, average_f1_base_legal, confusion_matrix_legal = test_model(legal_model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score, legal_bert=True)

    
        print("Accuracies: {} \n Average accuracy: {}".format(accuracies_base, average_accuracy_base))
        print("F1 Scores: {} \n Average F1: {}".format(f1_scores_base, average_f1_base))
    
        print("Accuracies Legal BERT: {} \n Average accuracy Legal BERT: {}".format(accuracies_base_legal, average_accuracy_base_legal))
        print("F1 Scores Legal BERT: {} \n Average F1 Legal BERT: {}".format(f1_scores_base_legal, average_f1_base_legal))

        
    result.append((parameters, (average_accuracy, average_f1)))     #add parameters and metrics to object for each configuration
        
    return result, confusion_matrix, running_lr, loss_over_epochs, model


def train_model_advanced(model, data_loader, loss_function, optimizer, scheduler, epochs, legal_bert=False):
    """
    Option to combine similar labels

    (explicit training function, called in larger train_test function)
    - grabs pre-trained sentence embeddings stored locally
    - trains model document by document across all 246 documents
    - averages final loss across all documents

    Parameters:
    - model : type of model used to train the data (BiLSTM or CNN_BiLSTM)
    - data_loader : function to gather embeddings
    - loss_function
    - optimizer
    - epochs : number of epochs you wish to train on


    Returns:
    - average loss across all documents (numpy float64 )
    """
    model.train()
    losses_over_epochs = []
    running_lr = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

#iterate over all documents
        for doc_idx in tqdm(range(246)): 
                if legal_bert:
                    TRAIN_emb = data_loader(filepath=f"train_document/doc_{doc_idx}_legal/embedding")
                    TRAIN_labels = data_loader(filepath=f"train_document/doc_{doc_idx}_legal/label")
                else:
                    TRAIN_emb = data_loader(filepath=f"train_document/doc_{doc_idx}/embedding")
                    TRAIN_labels = data_loader(filepath=f"train_document/doc_{doc_idx}/label")


                if TRAIN_emb.size(0) == 0: #ignore any faulty embeddings
                    continue
                output = model(TRAIN_emb) #push embeddings through model 
                loss = loss_function(output, TRAIN_labels) #calculate loss for document
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        print(f"Epoch: {epoch+1} | Document: {doc_idx+1}/246 | Loss: {loss.item():.5f}")
        running_lr.append(optimizer.state_dict()['param_groups'][0]['lr'])      #keep track of the variable learning rates over epochs

        losses_over_epochs.append(loss.item()) #store final loss for each document, then average across all documents
    return np.mean(losses_over_epochs), running_lr, losses_over_epochs, model




def test_model_advanced(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score, legal_bert=False):
    """
    Option to combine similar labels

    (explicit evaluation function, called in larger train_test function)
    - grabs test embeddings and labels for each test document
    - evaluates trained model and returns evaluation metrics

    Parameters:
    - model (class): type of model used to train the data (BiLSTM or CNN_BiLSTM)
    - data_loader (function): function to gather embeddings
    - calculate_confusion_matrix (function): function to build a confusion matrix with the given results
    - class_accuracy (function): function to calculate the accuracy with respect to a single class
    - class_f1_score (function):  function to calculate the f1-score with respect to a single class

    Returns:
    - accuracies (numpy array): accuracies for each individual class
    - f1_scores (numpy array): f1-scores for each individual class 
    - average_accuracy (numpy float64): average accuracy across all classes
    - average_f1 (numpy float64): average f1-score across all classes
    """
    confusion_matrix = None
    for i in range(29):
        if legal_bert:
            TEST_emb = data_loader(filepath=f"test_document/doc_{i}_legal/embedding")
            TEST_labels = data_loader(filepath=f"test_document/doc_{i}_legal/label")
        else:
            TEST_emb = data_loader(filepath=f"test_document/doc_{i}/embedding")
            TEST_labels = data_loader(filepath=f"test_document/doc_{i}/label")
            
        conf_matrix_helper = calculate_confusion_matrix(TEST_emb, TEST_labels, model)
        if confusion_matrix is None: #if no confusion matrix is given, use default confusion matrix
            confusion_matrix = conf_matrix_helper 
        else:
            confusion_matrix = np.add(confusion_matrix, conf_matrix_helper)
            
    accuracies = class_accuracy(confusion_matrix) #gather accuracies for each class from confusion matrix
    f1_scores = class_f1_score(confusion_matrix) #gather f1_scores for each class from confusion matrix
    average_accuracy = np.mean(accuracies)
    average_f1 = np.mean(f1_scores)

    return accuracies, f1_scores, average_accuracy, average_f1, confusion_matrix


def advanced_train_test(parameters, model, legal_model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score, get_class_weights):
    """
    (Overhead function called in main.py to train and evaluate model)
    
    Option to include the custom weighting for classes (to prevent the class-imbalance issue in the data)

    - train and test a model with a single set of parameters outlined in 'main.py'


    Parameters:
    - parameters (dictionary): hyperparameters given to the model (layers, hidden size, epochs etc.)
    - model (class): type of model used to train the data (BiLSTM or CNN_BiLSTM)
    - data_loader (function): function to gather embeddings
    - calculate_confusion_matrix (function): function to build a confusion matrix with the given results
    - class_accuracy (function): function to calculate the accuracy with respect to a single class
    - class_f1_score (function):  function to calculate the f1-score with respect to a single class
    - class_weights (pytorch tensor) : custom weights for classes to confront class-imbalance

    Returns:
    - result (list) : object display both the parameters used to train the model and evaluation metrics on how the model performed
    """
    class_weights = get_class_weights()
    result = []             #instantiate list object to hold info on parameters/evaluation
    #define model optimizer and loss function
    model_opt = torch.optim.Adam(model.parameters(), lr= parameters['learning_rate']) #define model optimizer and loss function
    legal_model_opt = torch.optim.Adam(legal_model.parameters(), lr= parameters['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_opt, T_max= parameters['epochs'],
                                                           eta_min= parameters['learning_rate_floor'])
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    print("\nWorking with: ")
    print(parameters)
    print(f"Model type: {model.__class__.__name__}")
    print("Train type: advanced")
    print("\n")

    print(f'{"Starting Training":-^100}')
    
    avg_loss, running_lr, loss_over_epochs, model = train_model_advanced(model, data_loader, loss_function, model_opt, scheduler, parameters['epochs'])
    
    avg_loss_legal, running_lr_legal, loss_over_epochs_legal, legal_model = train_model_advanced(legal_model, data_loader, loss_function, legal_model_opt, scheduler, parameters['epochs'], True)
    
    accuracies_base, f1_scores_base, average_accuracy_base, average_f1_base, confusion_matrix = test_model_advanced(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score)
    
    accuracies_base_legal, f1_scores_base_legal, average_accuracy_base_legal, average_f1_base_legal, confusion_matrix_legal = test_model(legal_model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score, legal_bert=True)

    
    print("Accuracies: {} \n Average accuracy: {}".format(accuracies_base, average_accuracy_base))
    print("F1 Scores: {} \n Average F1: {}".format(f1_scores_base, average_f1_base))
    
    print("Accuracies Legal BERT: {} \n Average accuracy Legal BERT: {}".format(accuracies_base_legal, average_accuracy_base_legal))
    print("F1 Scores Legal BERT: {} \n Average F1 Legal BERT: {}".format(f1_scores_base_legal, average_f1_base_legal))


    return result, confusion_matrix, running_lr, loss_over_epochs, model




def advanced_grid_search_train_test(parameters, model, legal_model, grid_search, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score, get_class_weights):
    """
    
    (Overhead function called in main.py to train and evaluate model)

    Option to include the custom weighting for classes (to prevent the class-imbalance issue in the data)

    - train and test multiple models using a grid search technique to test all possible input hyperparameters defined in 'main.py'

    Parameters:
    - parameters (dictionary): hyperparameters given to the model (layers, hidden size, epochs etc.)
    - model (class): type of model used to train the data (BiLSTM or CNN_BiLSTM)
    - data_loader (function): function to gather embeddings
    - calculate_confusion_matrix (function): function to build a confusion matrix with the given results
    - class_accuracy (function): function to calculate the accuracy with respect to a single class
    - class_f1_score (function):  function to calculate the f1-score with respect to a single class
    - class_weights (pytorch tensor) : custom weights for classes to confront class-imbalance


    Returns:
    - result (list) : object display both the parameters used to train the model and evaluation metrics on how the model performed
    - confusion matrix
    - running_lr
    - loss_over_epochs
    - model
    """
    class_weights = get_class_weights()


    result = []             #instantiate list object to hold info on parameters/evaluation

    parameter_configs = grid_search(parameters)         #gather parameters for the grid search defined in main.py
    
    #iterate over all possible hyperparameter configurations 
    for config in parameter_configs:

        model_opt = torch.optim.Adam(model.parameters(), lr= config['learning_rate'])       #define model optimizer and loss function
        legal_model_opt = torch.optim.Adam(legal_model.parameters(), lr= config['learning_rate'])

        loss_function = nn.CrossEntropyLoss(weight=class_weights)
        print("\nWorking with: ")
        print(config)
        print(f"Model type: {model.__class__.__name__}")
        print("Train type: grid search")

        print("\n")
        
        print(f'{"Starting Training":-^100}')
        # train_loss, running_lr, loss_over_epochs, model = train_model_advanced(model, data_loader, loss_function, model_opt, config['epochs'])         #train model
        
        # accuracies, f1_scores, average_accuracy, average_f1, confusion_matrix = test_model_advanced(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score)  #evaluate model, return metrics
        
        # print("Accuracies: {} \n Average accuracy: {}".format(accuracies, average_accuracy))
        # print("F1 Scores: {} \n Average F1: {}".format(f1_scores, average_f1))
        
        avg_loss, running_lr, loss_over_epochs, model = train_model_advanced(model, data_loader, loss_function, model_opt, config['epochs'])
    
        avg_loss_legal, running_lr_legal, loss_over_epochs_legal, legal_model = train_model_advanced(legal_model, data_loader, loss_function, legal_model_opt, config['epochs'], True)
    
        accuracies_base, f1_scores_base, average_accuracy_base, average_f1_base, confusion_matrix = test_model_advanced(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score)
    
        accuracies_base_legal, f1_scores_base_legal, average_accuracy_base_legal, average_f1_base_legal, confusion_matrix_legal = test_model_advanced(legal_model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score, legal_bert=True)

    
        print("Accuracies: {} \n Average accuracy: {}".format(accuracies_base, average_accuracy_base))
        print("F1 Scores: {} \n Average F1: {}".format(f1_scores_base, average_f1_base))
    
        print("Accuracies Legal BERT: {} \n Average accuracy Legal BERT: {}".format(accuracies_base_legal, average_accuracy_base_legal))
        print("F1 Scores Legal BERT: {} \n Average F1 Legal BERT: {}".format(f1_scores_base_legal, average_f1_base_legal))
        
    result.append((parameters, (average_accuracy, average_f1)))     #add parameters and metrics to object for each configuration
        
    return result, confusion_matrix, running_lr, loss_over_epochs, model


