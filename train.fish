for dataset in API_function_call Arithmetic_expression Array_usage Pointer_usage full
  echo $dataset
  set overrides "{'"train_data_path"': '"data/sysevr/$dataset/train"', "validation_data_path": '"data/sysevr/$dataset/val_test"', "test_data_path": '"data/sysevr/$dataset/test"'}"
  echo $overrides
  allennlp train configs/sysevr/cnn.jsonnet -s logs/sysevr/$dataset/cnn_128 -o $overrides
  allennlp train configs/sysevr/cnnh.jsonnet -s logs/sysevr/$dataset/cnnh_128 -o $overrides
  allennlp train configs/sysevr/lstm.jsonnet -s logs/sysevr/$dataset/lstm_1_128 -o $overrides
  allennlp train configs/sysevr/gru.jsonnet -s logs/sysevr/$dataset/gru_1_128 -o $overrides
  allennlp train configs/sysevr/lstm_attn.jsonnet -s logs/sysevr/$dataset/lstm_attn_1_128 -o $overrides
  allennlp train configs/sysevr/gru_attn.jsonnet -s logs/sysevr/$dataset/gru_attn_1_128 -o $overrides
end
