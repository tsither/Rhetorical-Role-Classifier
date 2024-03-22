import argparse
import time

from models import CNN_BiLSTM, BiLSTM

from utils import load_tensor, get_class_weights
from evaluation_functions import get_accuracy_value, grid_search, calculate_confusion_matrix, class_accuracy, class_f1_score
from helper_default import grid_search_train_test, default_train_test
from helper_advanced import advanced_train_test


###########################################################################

# main file, run this file in your command line with the arguments:

# --default_train OR --grid_search OR --advanced_train 
#        AND
# --bilstm OR --cnn_bilstm

###########################################################################

TRAIN_DATA_PATH = 'data/train.json'
TEST_DATA_PATH = 'data/dev.json'
 
remove_special_characters = False               #for data preprocessing 

def main():
    parser = argparse.ArgumentParser(
        description= "Train neural network for sequential sentence classification, generate word embeddings from scratch, or do both."
    )

    parser.add_argument(
        '--default_train', dest='default_train',
        help='Turn on this flag when you are ready to train the model',
        action='store_true'
    )

    parser.add_argument(
        '--grid_search', dest='grid_search',
        help='Train using grid search across multiple parameters',
        action='store_true'
    )
    parser.add_argument(
        '--advanced_train', dest='advanced_train',
        help='Train on single set of parameters with custom class weights & variable lr',
        action='store_true'
    )
    # parser.add_argument(
    #     '--advanced_grid_search', dest='advanced_grid_search',
    #     help='Train on single set of parameters with custom class weights & variable lr',
    #     action='store_true'
    # )
    
    parser.add_argument(
        '--bilstm', dest='bilstm',
        help='Use this flag to train a BiLSTM model',
        action='store_true'
    )
    
    parser.add_argument(
        '--cnn_bilstm', dest='cnn_bilstm',
        help='Use this flag to train a CNN_BiLSTM model',
        action='store_true'
    )

    args = parser.parse_args()


    if args.cnn_bilstm:
        model = CNN_BiLSTM()

    elif args.bilstm:
        model = BiLSTM()

    else:
        print("ERROR: No model chosen")

    #For training individual models
    parameters = {
        'epochs': 200,
        'learning_rate': 5e-4,
        'learning_rate_floor': 5e-5,
        'dropout': 0.25,
        'hidden_size': 256,
        'num_layers': 1
        }

    #For testing functionality:
    test_parameters = {
    'epochs': 1,
    'learning_rate': 0.0001,
    'dropout': 0.1,
    'hidden_size': 128,
    'num_layers': 1
    }


    #For grid search training
    parameter_configs = {
        'epochs': [10,20],
        'learning_rate': [0.0001, 0.001],
        'dropout': [0.0, 0.1, 0.2],
        'hidden_size': [128, 256],
        'num_layers': [1, 2]
        }

    
    if args.grid_search:
        result = grid_search_train_test(parameter_configs, 
                                        model=model, grid_search=grid_search, 
                                        data_loader=load_tensor, 
                                        calculate_confusion_matrix=calculate_confusion_matrix, 
                                        class_accuracy=class_accuracy, 
                                        class_f1_score=class_f1_score)

        max_accuracy_config = max(result, key=get_accuracy_value)

        print(max_accuracy_config)

    elif args.default_train:
        result = default_train_test(
            parameters=parameters,
            model=model,
            data_loader=load_tensor,
            calculate_confusion_matrix=calculate_confusion_matrix,
            class_accuracy=class_accuracy,
            class_f1_score=class_f1_score
        )
        max_accuracy_config = max(result, key=get_accuracy_value)

        print(max_accuracy_config)

    elif args.advanced_train:
        result = advanced_train_test(
                    parameters=parameters,
                    model=model,
                    data_loader=load_tensor,
                    calculate_confusion_matrix=calculate_confusion_matrix,
                    class_accuracy=class_accuracy,
                    class_f1_score=class_f1_score,
                    get_class_weights=get_class_weights
                )

    else:
        print("No model trained")





if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))