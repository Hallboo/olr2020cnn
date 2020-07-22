#!/bin/bash

step=1

dataset=/mipirepo/data/acoustic_scene_data/TAU-urban-acoustic-scenes-2019-mobile-development
wavefile=$dataset/audio
taskname=2019task1b
data=data/$taskname
featpath=$data/mono256dim

if [[ $step -eq 1 ]]; then
#   echo "## Extract Feature"
  python evaluate.py $dataset $featpath exp/$taskname || exit 1;
fi
