local batch_size = 32;
local num_epochs = 100;
local dataset = "API_function_call";
local train_data_path = "data/sysevr/%s/train" % [dataset];
local validation_data_path = "data/sysevr/%s/validation" % [dataset];
local test_data_path = "data/sysevr/%s/test" % [dataset];


// hyperparameters
local tokens_key = "tokens-hash";
local min_count = {"tokens": 3};
local embedding_dim = 64;
local seq2seq_input_dim = embedding_dim;
local num_layers = 1;
local feedforward_hidden_dim = 64;
local num_attention_heads = 8;
local positional_encoding = "sinusoidal";
local positional_embedding_size = 16;
local seq2seq_dropout = 0.1;
local activation = "relu";
local seq2vec_input_dim = embedding_dim;
local num_filters = 100;
local ngram_filter_sizes = [5, 6, 7, 8];
local dropout = 0.1;

//
local pretrain_type = "fasttext";
local pretrained_file = "data/sysevr/%s/embedding/%s_%s_%s.txt" % [dataset, pretrain_type, embedding_dim, tokens_key];

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
    "type": "seq2seq2vec",
    "embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": embedding_dim,
          "pretrained_file": pretrained_file
        }
      }
    },
    "seq2seq_encoder": {
      "type": "pytorch_transformer",
      "input_dim": seq2seq_input_dim,
      "num_layers": num_layers,
      "feedforward_hidden_dim": feedforward_hidden_dim,
      "num_attention_heads": num_attention_heads,
      "positional_encoding": positional_encoding,
      "positional_embedding_size": positional_embedding_size,
      "dropout_prob": seq2seq_dropout,
      "activation": activation
    },
    "seq2vec_encoder": {
      "type": "cnn",
      "embedding_dim": seq2vec_input_dim,
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
