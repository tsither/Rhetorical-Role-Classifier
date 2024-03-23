import numpy as np

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
    # if len(y_pred) != len(y_true):
    #     raise ValueError("Shapes of predictions and true labels must match. y_pred shape: {} y_true shape: {}".format(y_pred.shape, y_true.shape))

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    y_pred_np = y_pred.argmax(dim=1).cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    for pred, true in zip(y_pred_np, y_true_np):
        conf_matrix[pred, true] += 1

    return conf_matrix

def calculate_confusion_matrix(test_emb, test_labels, model, num_labels):
    model.eval()
    output = model(test_emb)
    return confusion_matrix(output, test_labels, num_labels)

def class_accuracy(conf_matrix):
    """
    Calculate accuracy for each class based on a confusion matrix.

    Parameters:
    conf_matrix (numpy.ndarray): Confusion matrix.

    Returns:
    list: List of accuracies for each class.
    """
    # diagonal = np.diag(conf_matrix)

    # row_sums = conf_matrix.sum(axis=1)

    # accuracies = diagonal / row_sums.astype(float)

    # return accuracies
    
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