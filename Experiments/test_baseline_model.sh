for dir in */;
  do
    cd $dir
    echo $dir
    python -u baseline_pnet_final.py
    cd ..
  done

