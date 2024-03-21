import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


###########################################################################

# File includes functions related to the training and evaluation process

###########################################################################


def train_model(model, data_loader, loss_function, optimizer, epochs):
    """
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
    batch_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

#iterate over all documents
        for doc_idx in tqdm(range(246)): 
                TRAIN_emb = data_loader(filepath=f"train_document/doc_{doc_idx}/embedding")
                TRAIN_labels = data_loader(filepath=f"train_document/doc_{doc_idx}/label")
                if TRAIN_emb.size(0) == 0: #ignore any faulty embeddings
                    continue
                output = model(TRAIN_emb) #push embeddings through model 
                loss = loss_function(output, TRAIN_labels) #calculate loss for document
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch: {epoch+1} | Document: {doc_idx+1}/246 | Loss: {loss.item():.5f}")
        batch_loss.append(loss.item()) #calculate loss across all documents
    return np.mean(batch_loss)



def test_model(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score):
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

    return accuracies, f1_scores, average_accuracy, average_f1


def default_train_test(parameters, model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score):
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
    loss_function = nn.CrossEntropyLoss()
    print("\nWorking with: ")
    print(parameters)
    print(f"Model type: {model.__class__.__name__}")
    print("Train type: default")
    print("\n")


    print(f'{"Starting Training":-^100}')
    train_loss = train_model(model, data_loader, loss_function, model_opt, parameters['epochs']) #train model
    
    #test model
    accuracies, f1_scores, average_accuracy, average_f1 = test_model(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score)
    print(f'accuracies {type(accuracies)}')

    print("Accuracies: {} \n Average accuracy: {}".format(accuracies, average_accuracy))
    print("F1 Scores: {} \n Average F1: {}".format(f1_scores, average_f1))
    
    result.append((parameters, (average_accuracy, average_f1)))


    return result




def grid_search_train_test(parameters, model, grid_search, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score):
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
        loss_function = nn.CrossEntropyLoss()
        print("\nWorking with: ")
        print(config)
        print(f"Model type: {model.__class__.__name__}")
        print("Train type: grid search")

        print("\n")

        
        
        print(f'{"Starting Training":-^100}')
        train_loss = train_model(model, data_loader, loss_function, model_opt, config['epochs'])         #train model
        
        accuracies, f1_scores, average_accuracy, average_f1 = test_model(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score)  #evaluate model, return metrics
        
        print("Accuracies: {} \n Average accuracy: {}".format(accuracies, average_accuracy))
        print("F1 Scores: {} \n Average F1: {}".format(f1_scores, average_f1))
        
    result.append((parameters, (average_accuracy, average_f1)))     #add parameters and metrics to object for each configuration
        
    return result










