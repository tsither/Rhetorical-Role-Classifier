# import torch
# import numpy as np

# def accuracy(y_pred, y_true):
#     """
#     Calculate accuracy of predictions.

#     Args:
#     y_pred (torch.Tensor): Predicted labels (batch_size, seq_len).
#     y_true (torch.Tensor): True labels (batch_size, seq_len).

#     Returns:
#     float: Accuracy score.
#     """
#     # Flatten and compare predictions with true labels
#     correct = (y_pred == y_true).sum().item()
#     total = y_pred.numel()
#     return correct / total

# def f1_score(y_pred, y_true):
#     """
#     Calculate F1 score of predictions.

#     Args:
#     y_pred (torch.Tensor): Predicted labels (batch_size, seq_len).
#     y_true (torch.Tensor): True labels (batch_size, seq_len).

#     Returns:
#     float: F1 score.
#     """
#     # Convert tensors to numpy arrays
#     y_pred_np = y_pred.cpu().numpy().flatten()
#     y_true_np = y_true.cpu().numpy().flatten()

#     # Calculate true positives, false positives, and false negatives
#     true_positives = np.sum(np.logical_and(y_pred_np == 1, y_true_np == 1))
#     false_positives = np.sum(np.logical_and(y_pred_np == 1, y_true_np == 0))
#     false_negatives = np.sum(np.logical_and(y_pred_np == 0, y_true_np == 1))

#     # Calculate precision, recall, and F1 score
#     precision = true_positives / (true_positives + false_positives + 1e-8)
#     recall = true_positives / (true_positives + false_negatives + 1e-8)
#     f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
#     return f1

# def evaluate_crf_model(model, data_loader):
#     """
#     Evaluate the CRF model on the given dataset.

#     Args:
#     model (torch.nn.Module): The CRF model to evaluate.
#     data_loader: DataLoader for the dataset.

#     Returns:
#     float: Average accuracy.
#     float: Average F1 score.
#     """
#     model.eval()
#     accuracies = []
#     f1_scores = []
#     model.eval()
#     with torch.no_grad():
#         for i in range(29):
#             TEST_emb = data_loader(filepath=f"test_document/doc_{i}/embedding")
#             TEST_labels = data_loader(filepath=f"test_document/doc_{i}/label")
#             # Forward pass
#             outputs = model(TEST_emb, TEST_labels)

#             # Convert outputs to predicted labels (0 or 1)
#             predicted = model.crf.decode()

#             # Calculate accuracy and F1 score
#             acc = accuracy(predicted, TEST_labels)
#             f1 = f1_score(predicted, TEST_labels)

#             accuracies.append(acc)
#             f1_scores.append(f1)

#     avg_accuracy = np.mean(accuracies)
#     avg_f1 = np.mean(f1_scores)

#     return avg_accuracy, avg_f1



import torch

def evaluate_crf_model(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_predicted = 0
    total_gold = 0

    with torch.no_grad():
        for i in range(29):
            TEST_emb = data_loader(filepath=f"test_document/doc_{i}/embedding")
            TEST_labels = data_loader(filepath=f"test_document/doc_{i}/label")
            # print(f"label shape before issue: {TEST_labels.shape}")
            

            rand_EMB = torch.rand(TEST_emb.shape)
            predicted_labels = model.predict(rand_EMB)

            
            predicted_labels = model.predict(TEST_emb)

            row_accuracies = []

    # Step 3: Iterate over each row and calculate accuracy
            for i in range(TEST_emb.shape[0]):
                # Calculate accuracy for each row
                accuracy = (TEST_emb[i] == TEST_labels[i]).sum().item() / TEST_emb.shape[1]
                row_accuracies.append(accuracy)


            # Compute metrics
            # correct = (predicted_labels == TEST_labels).sum().item()
            # total_correct += correct
            # total_predicted += predicted_labels.size(0) 
            # total_gold += TEST_labels.size(0)

            # for pred, gold in zip(predicted_labels, TEST_labels):

                # print(f"pred pred: {pred, type(pred)}")
                # print(f"gold shape: {gold.shape, type(gold)}")

                # total_correct += (pred == gold).sum().item()
                # total_predicted += pred.sum().item()  # Sum the lengths of individual predictions
                # total_gold += len(gold)

    # Compute accuracy
    print(f"len of row accuracies: {len(row_accuracies)}")
    # accuracy = total_correct / total_gold

    # # Compute F1-score
    # precision = total_correct / total_predicted
    # recall = total_correct / total_gold
    # f1_score = 2 * (precision * recall) / (precision + recall)
    pass
    # return accuracy
