#!/bin/bash

if [ $# -ne 2 ]; then
  echo -e "usage:\tbash hw5.sh [food_data_directory] [out_image_directory]"
  exit
fi

python3 draw_1.py $1 $2
python3 draw_2.py $1 $2
python3 draw_3.py $1 $2
python3 draw_4.py $1 $2

