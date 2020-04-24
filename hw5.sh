#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw5.sh [food_data_directory] [out_image_directory]"
  exit
fi

model=140.112.90.197:10297/hw5/model.ckpt

wget "${model}"

python3 draw_1.py $1 $2
python3 draw_2.py $1 $2
python3 draw_3.py $1 $2
python3 draw_4.py $1 $2

