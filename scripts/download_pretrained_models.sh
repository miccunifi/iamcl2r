#!/bin/bash

if [ -d "$1" ]; then
    DIR=$1
fi
if [ -z "$DIR" ]; then
    DIR='.'
fi

mkdir -p $DIR
FILENAME='pretrained_models_ckpts.zip'
wget -O ${DIR}'/pretrained_models_ckpts.zip'  'https://www.dropbox.com/scl/fi/fz2p1qkpwkd8nribcn9vd/pretrained_models_ckpts.zip?rlkey=w1nxxn2doxciq6dxun883n9du&dl=0'
unzip $DIR'/pretrained_models_ckpts.zip' -d $DIR
rm ${DIR}'/pretrained_models_ckpts.zip'

echo
echo
echo "The pretrained models have been downloaded into the $DIR directory."
if [ -d "$1" ]; then
    echo "Rembember to change the path to the pretrained models in the config file to $DIR"
fi