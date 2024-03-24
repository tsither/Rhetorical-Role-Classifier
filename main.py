import argparse
import time

from models import CNN_BiLSTM, BiLSTM

from utils import load_tensor, get_class_weights
from evaluation_functions import grid_search, calculate_confusion_matrix, class_accuracy, class_f1_score
from helper import grid_search_train_test, default_train_test, advanced_train_test, advanced_grid_search_train_test


###########################################################################

# main file, run this file in your command line with the arguments:

# --default_train OR --grid_search OR --advanced_train OR --advanced_grid_search
#        AND
# --bilstm OR --cnn_bilstm

###########################################################################

TRAIN_DATA_PATH = 'data/train.json'
TEST_DATA_PATH = 'data/dev.json'
 

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

    parser.add_argument(
        '--advanced_grid_search', dest='advanced_grid_search',
        help='Train on multiple sets of parameters with custom class weights & variable lr',
        action='store_true'
    )
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
        legal_model = CNN_BiLSTM()

    elif args.bilstm:
        model = BiLSTM()
        legal_model = BiLSTM()

    else:
        print("ERROR: No model chosen")

    #train individual models
    parameters = {
        'epochs': 50,
        'learning_rate': 5e-4,
        'learning_rate_floor': 5e-5,
        'dropout': 0.25,
        'hidden_size': 512,
        'num_layers': 2
        }

    #For testing functionality:
    test_parameters = {
    'epochs': 1,
    'learning_rate': 0.0001,
    'dropout': 0.1,
    'hidden_size': 128,
    'num_layers': 1
    }


    # For grid search training
    parameter_configs = {
        'epochs': [10,20],
        'learning_rate': [0.0001, 0.001],
        'dropout': [0.0, 0.1, 0.2],
        'hidden_size': [128, 256],
        'num_layers': [1, 2]
        }
    
    #testing grid search functionality
    test_parameter_configs = {
        'epochs': [1,2],
        'learning_rate': [0.0001],
        'dropout': [0.1],
        'hidden_size': [128],
        'num_layers': [1]
        }

    
    if args.grid_search:
        result = grid_search_train_test(test_parameter_configs, 
                                        model=model, legal_model=legal_model, grid_search=grid_search, 
                                        data_loader=load_tensor, 
                                        calculate_confusion_matrix=calculate_confusion_matrix, 
                                        class_accuracy=class_accuracy, 
                                        class_f1_score=class_f1_score)



    elif args.default_train:
        result = default_train_test(
            parameters=test_parameters,
            model=model,
            legal_model=legal_model,
            data_loader=load_tensor,
            calculate_confusion_matrix=calculate_confusion_matrix,
            class_accuracy=class_accuracy,
            class_f1_score=class_f1_score
        )


    elif args.advanced_train:
        result = advanced_train_test(
                    parameters=test_parameters,
                    model=model,
                    legal_model= legal_model,
                    data_loader=load_tensor,
                    calculate_confusion_matrix=calculate_confusion_matrix,
                    class_accuracy=class_accuracy,
                    class_f1_score=class_f1_score,
                    get_class_weights=get_class_weights
                )
        
    elif args.advanced_grid_search:
        result = advanced_grid_search_train_test(
                    parameters=test_parameter_configs,
                    model=model,
                    legal_model=legal_model,
                    grid_search=grid_search,
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