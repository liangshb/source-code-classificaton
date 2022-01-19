local batch_size = 32;
local num_epochs = 50;
local dataset = "Arithmetic_expression";
local train_data_path = "data/sysevr/%s/train" % [dataset];
local validation_data_path = "data/sysevr/%s/validation" % [dataset];
local test_data_path = "data/sysevr/%s/test" % [dataset];

// hyperparameters
local tokens_key = "merged-tokens-sym";
local min_count = {"tokens": 1};
local all_dim = std.parseInt(std.extVar('all_dim'));
local embedding_dim = all_dim;
local pretrained_file = "data/sysevr/%s/embedding/fasttext_%s_merged-tokens-sym.txt" % [dataset, embedding_dim];
local hidden_dim = all_dim;
local projection_dim = all_dim;
local feedforward_hidden_dim= all_dim;
local num_layers = std.parseInt(std.extVar('num_layers'));
local num_attention_heads = std.parseInt(std.extVar('num_attention_heads'));
local attn_dropout1 = std.parseJson(std.extVar('attn_dropout1'));
local attn_dropout2 = std.parseJson(std.extVar('attn_dropout2'));
local attn_dropout3 = std.parseJson(std.extVar('attn_dropout3'));
local num_filters = std.parseInt(std.extVar('num_filters'));
local ngram_filter_sizes = std.parseJson(std.extVar('ngram_filter_sizes'));
local num_highway = std.parseInt(std.extVar('num_highway'));
local projection_dim = all_dim;
local activation = "relu";
local projection_location = "after_highway";
local do_layer_norm = true;
local dropout = std.parseJson(std.extVar('dropout'));

// train
local lr = std.parseJson(std.extVar('lr'));

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
      "type": "cnn-highway-mask",
      "embedding_dim": hidden_dim,
      "num_filters": num_filters,
      "ngram_filter_sizes": ngram_filter_sizes,
      "num_highway": num_highway,
      "projection_dim": projection_dim,
      "activation": activation,
      "projection_location": projection_location,
      "do_layer_norm": do_layer_norm
    },
    "seq2seq_encoder": {
      "type": "stacked_self_attention",
      "input_dim": embedding_dim,
      "hidden_dim": hidden_dim,
      "projection_dim": projection_dim,
      "feedforward_hidden_dim": feedforward_hidden_dim,
      "num_layers": num_layers,
      "num_attention_heads": num_attention_heads,
      "dropout_prob": attn_dropout1,
      "residual_dropout_prob": attn_dropout2,
      "attention_dropout_prob": attn_dropout3
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
    "validation_metric": ["+mcc", "+f1", "+auc"],
    "optimizer": {
      "type": "adam",
      "lr": lr,
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 2
    },
    "callbacks": [
    {
      type: "optuna_pruner",
    }
    ]
  },
  "evaluate_on_test": true
}
