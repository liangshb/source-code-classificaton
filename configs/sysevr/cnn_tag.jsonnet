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
local tag_embedding_dim = 10;
local num_filters = 200;
local ngram_filter_sizes = [5, 6, 7, 8];
local dropout = 0.1;

// train
local lr = 0.002;
local weight_decay = 0.0005;

{
  "dataset_reader": {
    "type": "reader_tag",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "tokens"
      },
      "tags": {
        "type": "single_id",
        "namespace": "tags",
        "feature_name": "tag_"
      }
    },
    "token_key": token_key
  },
  "vocabulary": {
    "type": "from_instances",
    "min_count": min_count,
    "non_padded_namespaces": ["*labels"]
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
        },
        "tags": {
            "type": "embedding",
            "embedding_dim": tag_embedding_dim
        }
      }
    },
    "encoder": {
      "type": "cnn",
      "embedding_dim": embedding_dim + tag_embedding_dim,
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
