#!/bin/sh

python train.py -c 64input_depth2_x64.yml; python test.py -c 64input_depth2_x64.yml
python train.py -c 64input_depth2_x96.yml; python test.py -c 64input_depth2_x96.yml
python train.py -c 64input_depth3_x64.yml; python test.py -c 64input_depth3_x64.yml
python train.py -c 96input_depth2_x64.yml; python test.py -c 96input_depth2_x64.yml
python train.py -c 96input_depth2_x96.yml; python test.py -c 96input_depth2_x96.yml