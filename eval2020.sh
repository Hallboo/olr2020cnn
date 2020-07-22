#!/bin/bash

step=1

dataset=/mipirepo/data/acoustic_scene_data/TAU-urban-acoustic-scenes-2020-mobile-development
wavefile=$dataset/audio
taskname=2020task1a
data=data/$taskname/evaluation_setup
featpath=data/$taskname/mono256dim/norm

if [[ $step -eq 1 ]]; then
#   echo "## Extract Feature"
  python evaluate.py $data $featpath exp/$taskname/mono256dim || exit 1;
fi
