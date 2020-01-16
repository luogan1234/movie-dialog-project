for model in "pretrain" "vocab"; do
  for task in "genre"; do
    for vocab_dim in 16 32 48 64 80 96; do
      for t in {1..4}; do
        cmd="python main.py -model $model -task $task -vocab_dim $vocab_dim"
        echo $cmd
        $cmd
      done
    done
    for dialog_dim in 64 128 192 256 320 384; do
      for t in {1..4}; do
        cmd="python main.py -model $model -task $task -dialog_dim $dialog_dim"
        echo $cmd
        $cmd
      done
    done
    for feature_dim in 128 256 384 512 640 768; do
      for t in {1..4}; do
        cmd="python main.py -model $model -task $task -feature_dim $feature_dim"
        echo $cmd
        $cmd
      done
    done
  done
done
