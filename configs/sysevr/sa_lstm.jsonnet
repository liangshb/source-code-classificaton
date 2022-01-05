local batch_size = 128;
local num_epochs = 100;
local dataset = "API_function_call";
local train_data_path = "data/sysevr/%s/train" % [dataset];
local validation_data_path = "data/sysevr/%s/validation" % [dataset];
local test_data_path = "data/sysevr/%s/test" % [dataset];

// hyperparameters
local token_key = "tokens";
local min_count = {"tokens": 3};
local embedding_dim = 50;
local input_size = embedding_dim;
local hidden_size = 100;
local num_layers = 2;
local rnn_dropout = 0.1;
local use_highway = true;
local use_input_projection_bias = true;
local dropout = 0.1;

// train
local lr = 0.002;
local weight_decay = 0.0005;

{
  "dataset_reader": {
    "type": "reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "tokens"
      }
    },
    "token_key": token_key
  },
  "vocabulary": {
    "type": "from_instances",
    "min_count": min_count
  },
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,
  "test_data_path": test_data_path,
  "model": {
    "type": "classifier",
    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": embedding_dim
        }
      }
    },
    "encoder": {
      "type": "alternating_lstm",
      "input_size": input_size,
      "hidden_size": hidden_size,
      "num_layers": num_layers,
      "recurrent_dropout_probability": rnn_dropout,
      "use_highway": use_highway,
      "use_input_projection_bias": use_input_projection_bias
    },
    "dropout": dropout
  },
  "data_loader": {
    "batch_size": batch_size,
    "shuffle": true
  },
  "trainer": {
    "num_epochs": num_epochs,
    "patience": 10,
    "grad_norm": 5.0,
    "validation_metric": ["+mcc", "+auc", "+f1"],
    "optimizer": {
      "type": "adam",
      "lr": lr,
      "weight_decay": weight_decay,
      "betas": [0.9, 0.9]
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    }
  },
  "evaluate_on_test": true
}