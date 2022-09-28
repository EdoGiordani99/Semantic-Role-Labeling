# Model
MODEL_NAME = 'BERT_Model'
LANGUAGE_MODEL_NAME = "distilbert-base-uncased"
DROPOUT = 0.2
RNN_SIZE = 1024
NUM_RNN = 2
POS_RNN_SIZE = 128
POS_NUM_RNN = 2
BIDIRECTIONAL = True
BERT_FINE_TUNE = False

# Optimizer
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
BERT_LEARNING_RATE = 1e-5
BERT_WEIGHT_DECAY = 0.0
EPOCHS = 15

# DataLoaders
BATCH_SIZE = 16
NUM_WORKERS = 2
SHUFFLE = True

# Configuration
config = {'model_name': MODEL_NAME,
          'language_model_name': LANGUAGE_MODEL_NAME,
          'dropout': DROPOUT,
          'rnn_size': RNN_SIZE,
          'bidirectional': BIDIRECTIONAL,
          'num_rnn': NUM_RNN,
          'bert_fine_tune': BERT_FINE_TUNE,
          'learning_rate': LEARNING_RATE,
          'weight_decay': WEIGHT_DECAY,
          'bert_learning_rate': BERT_LEARNING_RATE,
          'bert_weight_decay': BERT_WEIGHT_DECAY,
          'epochs': EPOCHS,
          'batch_size': BATCH_SIZE,
          'num_workers': NUM_WORKERS,
          'shuffle': SHUFFLE}
