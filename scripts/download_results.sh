#!/bin/bash

if [ -d "$1" ]; then
    DIR=$1
fi
if [ -z "$DIR" ]; then
    echo "The results.zip file will be downloaded into the notebooks directory."
    DIR='./notebooks'
fi

mkdir -p $DIR
wget -O $DIR'/results.zip' 'https://www.dropbox.com/scl/fi/ddf55n2dk7cjwsiicz245/output.zip?rlkey=u77m9ew80isztdrwyf6x9heq6&dl=0'
unzip -o $DIR'/results.zip' -d $DIR
rm $DIR'/results.zip'

echo
echo
echo "The results.zip file has been downloaded into the $DIR directory."
if [ -d "$1" ]; then
    echo "Rembember to change the path to the results in the eval config file to $DIR"
fi