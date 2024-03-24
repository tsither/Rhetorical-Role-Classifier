import numpy as np
import torch
from itertools import product


###########################################################################

# file storing the functions relevant for evaluation

###########################################################################




def test_accuracy(x_test, y_test, model):
    """

    Parameters:

    Returns:
    """
    model.eval()
    output = model(x_test)
    acc = sum(output.argmax(dim=1) == y_test)/ output.size(0)
    print(f"Test Accuracy {acc*100:.2f}%")
    return acc*100


def set_highest_to_one(tensor):
    """

    Parameters:

    Returns:
    """
    max_val, max_idx = tensor.max(dim=1, keepdim=True)
    result = torch.zeros_like(tensor)
    result.scatter_(1, max_idx, 1)
    return result


def class_accuracy(conf_matrix):
    """
    Calculate accuracy for each class based on a confusion matrix.

    Parameters:
    conf_matrix (numpy array): Confusion matrix.

    Returns:
    list: List of accuracies for each class.
    """
    
    diagonal = np.diag(conf_matrix)
    row_sums = conf_matrix.sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        accuracies = np.where(row_sums != 0, diagonal / row_sums.astype(float), 0.0)

    return accuracies



def class_f1_score(conf_matrix, epsilon=1e-7):
    """
    Calculate F1 score for each class based on a confusion matrix.

    Parameters:
    conf_matrix (numpy.ndarray): Confusion matrix.
    epsilon (float): Smoothing term to avoid division by zero.

    Returns:
    list: List of F1 scores for each class.
    """
    
    tp = np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1_scores = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1_scores


def confusion_matrix(y_pred, y_true, num_classes):
    """
    Create a confusion matrix for label encodings in PyTorch.

    Parameters:
    y_pred (torch.Tensor): Predicted labels tensor.
    y_true (torch.Tensor): True labels tensor.
    num_classes (int): Number of classes.

    Returns:
    numpy.ndarray: Confusion matrix.
    """ 
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    y_pred_np = y_pred.argmax(dim=1).cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    for pred, true in zip(y_pred_np, y_true_np):
        conf_matrix[pred, true] += 1

    return conf_matrix


def grid_search(parameters):
    """

    Parameters:

    Returns:
    """
    keys = parameters.keys()
    values = parameters.values()
    
    combinations = list(product(*values))
    
    parameter_configurations = [{k: v for k, v in zip(keys, combination)} for combination in combinations]
    
    return parameter_configurations


def calculate_confusion_matrix(test_emb, test_labels, model):
    """

    Parameters:

    Returns:
    """
    model.eval()
    output = model(test_emb)
    return confusion_matrix(output, test_labels, 13)

def get_accuracy_value(item):
    """

    Parameters:

    Returns:
    """
    return item[1][0]