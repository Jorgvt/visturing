for dir in */;
  do
    cd $dir
    # cp param_pnet_final.py param_pnet_final_split.py
    cp param_pnet_final.py baseline_pnet_final.py
    cd ..
  done

