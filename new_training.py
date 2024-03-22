import torch
import torch.nn as nn
from Dataset_Reader import Dataset_Reader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# from utils import get_class_weights 
from utils import load_tensor
from models import BiLSTM
# from helper_default import gri


TRAIN_DATA_PATH = 'data/train.json'
TEST_DATA_PATH = 'data/dev.json'

data_train = Dataset_Reader(TRAIN_DATA_PATH)
data_test = Dataset_Reader(TEST_DATA_PATH)


list_of_targets = ['ISSUE', 'FAC', 'NONE', 'ARG_PETITIONER', 'PRE_NOT_RELIED', 'STA', 'RPC', 'ARG_RESPONDENT', 'PREAMBLE', 'ANALYSIS', 'RLC', 'PRE_RELIED', 'RATIO'] #rename to: list_unique_targets
total_encoded_unique_targets = LabelEncoder().fit(list_of_targets)    #label_encoder1

combined_unique_targets = list(set(data_train.labels)) #label_encoder #includes the combination of the different types of arg and pre labels         
combined_encoded_unique_targets = LabelEncoder().fit(combined_unique_targets)


def remap_targets(target_tensor: torch.TensorType, old_le, new_le):
    """
    - Remap targets from label encoder that contains all classes to new label encoder that combines similar labels 
    e.g. ('ARG_PETITIONER' + 'ARG_RESPONDENT') --> 'ARG' & ('PRE_NOT_RELIED' + 'PRE_RELIED') --> 'PRE'

    Parameters: 
    - target tensor (pytorch tensor) : tensor populated with labels from data
    - old_le (label encoder class) : populated label encoder class containing all labels
    - new_le (label encoder class) : populated label encoder class containing combined labels

    Returns:
    new_tensor (pytorch tensor) : tensor populated with new combined labels where relevant
    """
    inverse_tensor = old_le.inverse_transform(target_tensor.long())
    for idx, label in enumerate(inverse_tensor):
        if label == 'ARG_RESPONDENT' or label == 'ARG_PETITIONER':
            inverse_tensor[idx] = "ARG"
        elif label == 'PRE_NOT_RELIED' or label == 'PRE_RELIED':
            inverse_tensor[idx] = 'PRE'
    new_tensor = torch.tensor(new_le.transform(inverse_tensor))
    return new_tensor




def get_class_weights(y_train) -> torch.FloatTensor:
    class_weights = compute_class_weight(class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train)
    class_weights = torch.FloatTensor(class_weights)
    return class_weights


sample_input, sample_target = None, None
for idx in range(246):
    if sample_input is None:
        sample_input = load_tensor(filepath=f"../train_document/doc_{idx}/embedding")
        sample_target = load_tensor(filepath=f"../train_document/doc_{idx}/label")
    else:
        sample_input = torch.cat((sample_input,load_tensor(filepath=f"../train_document/doc_{idx}/embedding")), dim=0)
        sample_target = torch.cat((sample_target,load_tensor(filepath=f"../train_document/doc_{idx}/label")), dim=0)

sample_target = sample_target.long() 
class_weights = get_class_weights(sample_target.numpy())




def uniform_sample(input_tensor, target_tensor, num_classes):
    class_indices = [np.where(target_tensor.numpy() == i)[0] for i in range(num_classes)]
    min_class_samples = min(len(indices) for indices in class_indices)
    print([len(indices) for indices in class_indices]) 
    print(min_class_samples)
    sampled_indices = []
    for indices in class_indices:
        sampled_indices.extend(np.random.choice(indices, min_class_samples, replace=False))
    
    sampled_indices = np.random.permutation(sampled_indices)
    # print(sampled_indices)
    
    return input_tensor[sampled_indices], target_tensor[sampled_indices]















#NEXT
# def grid_search_train_test(parameters, class_weights = None):
#     result = []
    
#     parameter_configs = grid_search(parameters)
#     for config in parameter_configs:
#         loss_list = []
#         running_lr = []
#         model = BiLSTM(hidden_size=config['hidden_size'], num_layers=config['num_layers'], dropout=config['dropout'], output_size= 11)
#         model_opt = torch.optim.Adam(model.parameters(), lr= config['learning_rate'])
#         # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_opt, T_max= config['epochs'], eta_min= config['learning_rate_floor'])
#         # scheduler1 = torch.optim.lr_scheduler.ConstantLR(model_opt, factor= 0.8, total_iters= config['epochs']*0.4)
#         # scheduler2 = torch.optim.lr_scheduler.ConstantLR(model_opt, factor= 0.6, total_iters= config['epochs']*0.7)
#         loss_function = nn.CrossEntropyLoss(weight=class_weights)
#         print("Working with: ")
#         print(config)
#         print(f'{"Starting Training":-^100}')
#         model.train()
#         for epoch in tqdm(range(config['epochs'])):
#             running_loss = []
#             for idx in range(246):
#                 TRAIN_emb = load_tensor(filepath=f"../train_document/doc_{idx}/embedding")
#                 TRAIN_labels = load_tensor(filepath=f"../train_document/doc_{idx}/label")
#                 TRAIN_labels = remap_targets(TRAIN_labels, label_encoder1, label_encoder)
#                 if TRAIN_emb.size(0) == 0:
#                     continue
#                 output = model(TRAIN_emb)
#                 loss = loss_function(output,TRAIN_labels)
                
#                 model_opt.zero_grad()
#                 loss.backward()
#                 model_opt.step()
#                 running_loss.append(loss.item())
#             # scheduler.step()
#             # scheduler1.step()
#             # scheduler2.step()
#             running_lr.append(model_opt.state_dict()['param_groups'][0]['lr'])
#             loss_list.append(np.mean(running_loss))
#             print(f"Epoch: {epoch+1} \t Loss: {np.mean(running_loss):.5f} \t LR: {model_opt.state_dict()['param_groups'][0]['lr']}")
#         # batch_loss.append(loss.item())
#         cm = None
#         for i in range(29):
#             TEST_emb = load_tensor(filepath=f"../test_document/doc_{i}/embedding")
#             TEST_labels = load_tensor(filepath=f"../test_document/doc_{i}/label")
#             TEST_labels = remap_targets(TEST_labels, label_encoder1, label_encoder)
#             conf_matrix_helper = calculate_confusion_matrix(TEST_emb, TEST_labels, model, num_labels= 11)
#             if cm is None:
#                 cm = conf_matrix_helper
#             else:
#                 cm = np.add(cm, conf_matrix_helper)
                
#         accuracies = class_accuracy(cm)
#         f1_scores = class_f1_score(cm)
#         average_accuracy = np.mean(accuracies)
#         average_f1 = np.mean(f1_scores)

#         print("Accuracies: {} \n Average acccuracy: {}".format(accuracies, average_accuracy))
#         print("F1 Scores: {} \n Average F1: {}".format(f1_scores, average_f1))
#         result.append((config, (average_accuracy, average_f1)))
#     return cm, loss_list, result, running_lr, model

