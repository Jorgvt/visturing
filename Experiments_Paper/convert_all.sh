for dir in */;
  do
    cd $dir
    jupyter nbconvert --to python --no-prompt *.ipynb
    cd ..
  done

