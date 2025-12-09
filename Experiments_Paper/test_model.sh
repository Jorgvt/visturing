for dir in */;
  do
    cd $dir
    echo $dir
    python -u param_pnet_final.py --model-name $1
    cd ..
  done

