local batch_size = 256;
local num_epochs = 100;
local dataset = "Array_usage";
local train_data_path = "data/sysevr/%s/train" % [dataset];
local validation_data_path = "data/sysevr/%s/val_test" % [dataset];
local test_data_path = "data/sysevr/%s/test" % [dataset];

// hyperparameters
local tokens_key = "merged-tokens-sym";
local min_count = {"tokens": 1};
local embedding_dim = 64;
local num_filters = 50;
local ngram_filter_sizes = [5, 6, 7, 8];
local num_highway = 1;
local projection_dim = 50;
local activation = "relu";
local projection_location = "after_highway";
local do_layer_norm = true;
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
    "tokens_key": tokens_key
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
      "type": "cnn-highway-mask",
      "embedding_dim": embedding_dim,
      "num_filters": num_filters,
      "ngram_filter_sizes": ngram_filter_sizes,
      "num_highway": num_highway,
      "projection_dim": projection_dim,
      "activation": activation,
      "projection_location": projection_location,
      "do_layer_norm": do_layer_norm
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
    },
    "callbacks": [
      {
        "type": "tensorboard"
      }
    ]
  },
  "evaluate_on_test": true
}
