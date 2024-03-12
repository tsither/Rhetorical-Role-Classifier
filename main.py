import argparse
import time

from models import CNN_BiLSTM, BiLSTM

from utils import load_tensor
from evaluation_functions import get_accuracy_value, grid_search, calculate_confusion_matrix, class_accuracy, class_f1_score
from helper import grid_search_train_test, default_train_test


###########################################################################

# main file, run this file in your command line with the arguments:

# --train OR --grid_search
#        AND
# --bilstm OR --cnn_bilstm

###########################################################################


def main():
    parser = argparse.ArgumentParser(
        description= "Train neural network for sequential sentence classification"
    )

    parser.add_argument(
        '--train', dest='train',
        help='Turn on this flag when you are ready to train the model',
        action='store_true'
    )

    
    parser.add_argument(
        '--grid_search', dest='grid_search',
        help='Train using grid search across multiple parameters',
        action='store_true'
    )
    
    parser.add_argument(
        '--bilstm', dest='bilstm',
        help='Use this flag to train a BiLSTM model (default)',
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

    else:
        model = BiLSTM()


    #For default training:
    parameters = {
    'epochs': 1,
    'learning_rate': 0.0001,
    'dropout': 0.1,
    'hidden_size': 128,
    'num_layers': 1
    }


    #For grid search training
    parameter_configs = {
        'epochs': [10,20,30],
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
    else:
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





if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))