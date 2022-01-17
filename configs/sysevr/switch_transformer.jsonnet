local batch_size = 64;
local num_epochs = 100;
local dataset = "Arithmetic_expression";
local train_data_path = "data/sysevr/%s/train" % [dataset];
local validation_data_path = "data/sysevr/%s/val_test" % [dataset];
local test_data_path = "data/sysevr/%s/test" % [dataset];

// hyperparameters
local tokens_key = "merged-tokens-sym";
local min_count = {"tokens": 1};
local embedding_dim = 64;
local capacity_factor = 1.0;
local drop_tokens = true;
local is_scale_prob = true;
local n_experts = 10;
local num_heads = 8;
local ff_dim = 64;
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
    "tokens_key": tokens_key,
  },
  "vocabulary": {
    "type": "from_instances",
    "min_count": min_count
  },
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,
  "test_data_path": test_data_path,
  "model": {
    "type": "classifier-plus",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": embedding_dim
        }
      }
    },
    "seq2vec_encoder": {
      "type": "boe",
      "embedding_dim": embedding_dim,
      "averaged": true
    },
    "seq2seq_encoder": {
      "type": "switch-transformer",
      "layer": {
        "input_dim": embedding_dim,
        "attn": {
          "num_heads": 4,
          "input_dim": 64,
          "dropout": 0.1
        },
        "feed_forward": {
          "capacity_factor": 1.0,
          "drop_tokens": true,
          "is_scale_prob": true,
          "n_experts": 10,
          "expert": {
            "input_dim": embedding_dim,
            "hidden_dim": embedding_dim,
            "dropout": 0.1
          },
          d_model: embedding_dim
        },
        dropout1: 0.1,
        dropout2: 0.1
      },
      "n_layers": 1,
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
