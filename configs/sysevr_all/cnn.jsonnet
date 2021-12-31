local batch_size = 64;
local num_epochs = 100;
local train_data_path = "data/sysevr/tokenized/train";
local validation_data_path = "data/sysevr/tokenized/validation";
local test_data_path = "data/sysevr/tokenized/test";

// hyperparameters
local min_count = {"tokens": 5};
local embedding_dim = 50;
local pretrain_type = "fasttext";
local pretrained_file = "data/sysevr/embedding/%s_%s.txt" % [pretrain_type, embedding_dim];
local dropout = 0.1;
local lr = 0.002;
local weight_decay = 0.0005;
local num_filters = 200;
local ngram_filter_sizes = [4, 5, 6];

{
  "dataset_reader": {
    "type": "reader",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "tokens"
      }
    }
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
      "type": "cnn",
      "embedding_dim": embedding_dim,
      "num_filters": num_filters,
      "ngram_filter_sizes": ngram_filter_sizes,
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
    "validation_metric": ["+mcc","+auc", "+f1"],
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
