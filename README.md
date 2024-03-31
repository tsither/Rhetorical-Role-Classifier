# Lawfluence - LegalEval Subtask A
This project is an implementation of the Sem-Eval 2023 LegalEval Subtask A, submitted as a final project for the course Wi23/24 Advanced Natural Language Processing at the University of Potsdam. 

## 1. Task
The task outlined by Sem-Eval 2023 is as follows:
Given an annotated corpus of legal documents with which to train, predict each sentence in a set of documents as corresponding to a set of 13 distinct Rhetorical Role (RR) labels. Rhetorical roles are relevant sentence categories present in judgement texts. All 13 RR labels can be found in the appendix.

## 2. Data
The data is comprised of Indian court judgement texts. The documents were annotated at the sentence level with respect to 13 distinct RRs.  

### 2.1 Input Data Format

The top level structure of each JSON file is a list, where each entry represents a judgement-labels data point. Each data point is
a dict with the following keys:
- `id`: a unique id for this  data point. This is useful for evaluation.
- `annotations`:list of dict.The items in the dict are:
  - `result`a list of dictionaries containing sentence text and corresponding labels pair.The keys are:
    - `id`:unique id of each sentence
    - `value`:a dictionary with the following keys:
      - `start`:integer.starting index of the text
      - `end`:integer.end index of the text
      - `text`:string.The actual text of the sentence
      - `labels`:list.the labels that correspond to the text
- `data`: the actual text of the judgement.
- `meta`: a string.It tells about the category of the case(Criminal,Tax etc.)


## 3. Training the model 

Training (and evaluating) the various models in this project can be done via the following flags on your terminal (assuming you are in the projects directory). Dependencies required to run this project can be found in 'requirements.txt'.


### 3.1 Model flags
Choose one of the following flags to denote model architecture:

```bash
--bilstm OR --cnn_bilstm
```

### 3.2 Training flags
Choose one of the following flags to denote training:

```bash
--default_train OR --advanced_train OR --grid_search OR --advanced_grid_search
```
'--default_train' marks the standard training process with standard BERT embeddings

'--advanced_train' trains the model on (separately) standard BERT and LegalBERT embeddings, and applies custom class weights to the classes (tool for approaching the class imbalance issue in dataset)

'--grid_search' trains the model using grid search functionality in default mode (standard BERT embeddings). The model will train iteratively over different sets of hyper-parameters. 

'--advanced_grid_search' trains the model using grid search functionality with the advanced training feature (both standard BERT and legalBERT embeddings + custom class weights). The model will train iteratively over different sets of hyper-parameters. 


The parameters used for training can be found in 'main.py' on lines 79-95.

```bash
legaleval-subtask-a-main % python main.py --(model flag) --(training flag)
```
For example:

```bash
legaleval-subtask-a-main % python main.py --bilstm --advanced_train
```
## 4. Evaluating performance

### 4.1 Hyperparameters
The hyperparameters for the current run are displayed immediately after execution:

```bash
Working with: 
{'epochs': 100, 'learning_rate': 0.0005, 'learning_rate_floor': 5e-06, 'dropout': 0.25, 
'hidden_size': 512, 'num_layers': 1}
Model type: BiLSTM
Train type: advanced
```

### 4.2 Evaluation metrics
Once the training is complete, the relevant accuracies and F1-scores will printed, along with the accuracies and F1-scores for each class.
For example: 

```bash
Accuracies: [0.81181619 0.33333333 0.42105263 0.82233503 0.74509804 0.95604396
 0.99204771 0.         0.47311828 0.52054795 0.53246753 0.82178218
 0.625     ] 

 Average accuracy (standard BERT): 0.619587909791633

F1 Scores: [0.7856008  0.26548668 0.29090904 0.83076918 0.7524752  0.93548382
 0.99106251 0.         0.63007155 0.52413788 0.42487042 0.87830683
 0.56603768] 
 Average F1: 0.6057855070889212

Accuracies Legal BERT: [0.67161227 0.04761905 0.04761905 0.49598163 0.36842105 0.45238095
 0.81405896 0.         0.25757576 0.38636364 0.125      0.47058824
 0.09615385] 

 Average accuracy Legal BERT: 0.32564418676524204

F1 Scores Legal BERT: [0.68378646 0.02247187 0.06060601 0.59586202 0.31818177 0.16379307
 0.75978831 0.         0.24999995 0.2931034  0.01612902 0.57142852
 0.12345674] 
 Average F1 Legal BERT: 0.2968159349436752
```
*Note that because of the '--advanced_train' flag, two distinct training runs (one with standard BERT embeddings, the other with legalBERT embeddings) were executed. 

## 5. Generating the pre-trained embeddings
We used two different pre-trained models in this project:
1) BERT ('bert-base-uncased') 
2) LegalBERT ('nlpaueb/legal-bert-base-uncased')

The training of these models on this data was conducted in the file 'emb_label_generation.py'. Training is executed simply by running this file: 
```bash
legaleval-subtask-a-main % python emb_label_generation.py
```

## Appendix

Rhetorical roles (RRs): 
- Preamble (PREAMBLE)
- Facts (FAC)
- Ruling by Lower Court (RLC)
- Issues (ISSUE)
- Argument by Petitioner (ARG_PETITIONER)
- Argument by Respondent (ARG_RESPONDENT)
- Analysis (ANALYSIS)
- Statute (STA)
- Precendent Relied (PRE_RELIED)
- Precendent Not Relied (PRE_NOT_RELIED)
- Ratio of the decision (RATIO)
- Ruling by Present Court (RPC)
- NONE

