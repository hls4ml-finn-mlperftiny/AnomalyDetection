#!/bin/sh

python train.py -c 84input_depth2_x64_10b.yml; python test.py -c 84input_depth2_x64_10b.yml
python train.py -c 84input_depth2_x64_12b.yml; python test.py -c 84input_depth2_x64_12b.yml
python train.py -c 84input_depth2_x64_16b.yml; python test.py -c 84input_depth2_x64_16b.yml
python train.py -c 84input_depth2_x72_8b.yml; python test.py -c 84input_depth2_x72_8b.yml
python train.py -c 84input_depth2_x72_10b.yml; python test.py -c 84input_depth2_x72_10b.yml
python train.py -c 84input_depth2_x72_12b.yml; python test.py -c 84input_depth2_x72_12b.yml
python train.py -c 84input_depth2_x72_14b.yml; python test.py -c 84input_depth2_x72_14b.yml