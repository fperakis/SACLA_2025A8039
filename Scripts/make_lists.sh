#!/bin/bash

#PBS -N maketagsandconvert

ml purge
ml MakeTagList/latest
ml DataConvert4

MakeTagList -b 2 -r ${1} -out /UserData/andronis/test_data/processed/tags/${1}.lst
DataConvert4 -l /UserData/andronis/test_data/processed/tags/${1}.lst -dir /UserData/andronis/test_data/processed/converted -o ${1}.h5
