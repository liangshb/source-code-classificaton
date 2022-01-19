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
local pretrained_file = "data/sysevr/%s/embedding/fasttext_%s_merged-tokens-sym.txt" % [dataset, embedding_dim];
local num_layers = 1;
local rnn_dropout = 0.1;
local bidirectional = true;
local num_filters = 200;
local ngram_filter_sizes = [5, 6, 7, 8];
local num_highway = 2;
local projection_dim = 64;
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
          "embedding_dim": embedding_dim,
          "pretrained_file": pretrained_file
        }
      }
    },
    "seq2vec_encoder": {
      "type": "boe",
      "embedding_dim": embedding_dim,
      "averaged": true
    },
    "seq2seq_encoder": {
      "type": "qanet_encoder",
      "input_dim": embedding_dim,
      "hidden_dim": embedding_dim,
      "attention_projection_dim": embedding_dim,
      "feedforward_hidden_dim": embedding_dim,
      "num_blocks": 1,
      "num_convs_per_block": 2,
      "conv_kernel_size": 5,
      "num_attention_heads": 4,
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
