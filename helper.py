import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


###########################################################################

# File includes functions related to the training and evaluation process

###########################################################################




def train_model(model, data_loader, loss_function, optimizer, epochs):
    model.train()
    batch_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        for doc_idx in tqdm(range(246)):
                TRAIN_emb = data_loader(filepath=f"train_document/doc_{doc_idx}/embedding")
                TRAIN_labels = data_loader(filepath=f"train_document/doc_{doc_idx}/label")
                if TRAIN_emb.size(0) == 0:
                    continue
                output = model(TRAIN_emb)
                loss = loss_function(output, TRAIN_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch: {epoch+1} | Document: {doc_idx+1}/246 | Loss: {loss.item():.5f}")

        batch_loss.append(loss.item())
    return np.mean(batch_loss)



def test_model(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score):
        confusion_matrix = None
        for i in range(29):
            TEST_emb = data_loader(filepath=f"test_document/doc_{i}/embedding")
            TEST_labels = data_loader(filepath=f"test_document/doc_{i}/label")
            conf_matrix_helper = calculate_confusion_matrix(TEST_emb, TEST_labels, model)
            if confusion_matrix is None:
                confusion_matrix = conf_matrix_helper
            else:
                confusion_matrix = np.add(confusion_matrix, conf_matrix_helper)
                
        accuracies = class_accuracy(confusion_matrix)
        f1_scores = class_f1_score(confusion_matrix)
        average_accuracy = np.mean(accuracies)
        average_f1 = np.mean(f1_scores)
        return accuracies, f1_scores, average_accuracy, average_f1


def default_train_test(parameters, model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score):
    result = []
    model_opt = torch.optim.Adam(model.parameters(), lr= parameters['learning_rate'])
    loss_function = nn.CrossEntropyLoss()
    print("\nWorking with: ")
    print(parameters)
    print(f"Model type: {model.__class__.__name__}")
    print("Train type: default")
    print("\n")


    print(f'{"Starting Training":-^100}')
    train_loss = train_model(model, data_loader, loss_function, model_opt, parameters['epochs'])
    
    accuracies, f1_scores, average_accuracy, average_f1 = test_model(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score)
    
    print("Accuracies: {} \n Average accuracy: {}".format(accuracies, average_accuracy))
    print("F1 Scores: {} \n Average F1: {}".format(f1_scores, average_f1))
    
    result.append((parameters, (average_accuracy, average_f1)))


    return result




def grid_search_train_test(parameters, model, grid_search, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score):
    result = []
    parameter_configs = grid_search(parameters)
    
    for config in parameter_configs:

        model_opt = torch.optim.Adam(model.parameters(), lr= config['learning_rate'])
        loss_function = nn.CrossEntropyLoss()
        print("\nWorking with: ")
        print(config)
        print(f"Model type: {model.__class__.__name__}")
        print("Train type: grid search")

        print("\n")

        
        
        print(f'{"Starting Training":-^100}')
        train_loss = train_model(model, data_loader, loss_function, model_opt, config['epochs'])
        
        accuracies, f1_scores, average_accuracy, average_f1 = test_model(model, data_loader, calculate_confusion_matrix, class_accuracy, class_f1_score)
        
        print("Accuracies: {} \n Average accuracy: {}".format(accuracies, average_accuracy))
        print("F1 Scores: {} \n Average F1: {}".format(f1_scores, average_f1))
        
        result.append((config, (average_accuracy, average_f1)))
        
    return result












