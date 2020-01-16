for model in "pretrain" "vocab"; do
  for t in {1..1}; do
    for task in "IMDB" "gender"; do
      cmd="python ml_main.py -model $model -task $task"
      echo $cmd
      $cmd
    done
    for task in "genre"; do
      for i in {0..23}; do
        cmd="python ml_main.py -model $model -task $task -genre_id $i"
        echo $cmd
        $cmd
      done
    done
    for task in "IMDB"; do
      cmd="python ml_main.py -model $model -task $task -regression"
      echo $cmd
      $cmd
    done
  done
done
