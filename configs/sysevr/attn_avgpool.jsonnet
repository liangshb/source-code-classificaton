local batch_size = 64;
local num_epochs = 100;
local dataset = "Array_usage";
local train_data_path = "data/sysevr/%s/train" % [dataset];
local validation_data_path = "data/sysevr/%s/val_test" % [dataset];
local test_data_path = "data/sysevr/%s/test" % [dataset];

// hyperparameters
local tokens_key = "merged-tokens-sym";
local min_count = {"tokens": 1};
local embedding_dim = 64;
local pretrained_file = "data/sysevr/%s/embedding/fasttext_%s_merged-tokens-sym.txt" % [dataset, embedding_dim];
local hidden_dim = 64;
local projection_dim = 64;
local feedforward_hidden_dim= 64;
local num_layers = 1;
local num_attention_heads = 4;
local attn_dropout1 = 0.1;
local attn_dropout2 = 0.2;
local attn_dropout3 = 0.1;
local dropout = 0.1;
local averaged = true;

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
      "embedding_dim": hidden_dim,
      "averaged": averaged
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
