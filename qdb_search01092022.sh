#!/bin/sh

python train.py -c qdb_input96_depth2_x64_8b.yml; python test.py -c qdb_input96_depth2_x64_8b.yml
python train.py -c qdb_input96_depth2_x64_10b.yml; python test.py -c qdb_input96_depth2_x64_10b.yml
python train.py -c qdb_input96_depth2_x64_12b.yml; python test.py -c qdb_input96_depth2_x64_12b.yml
python train.py -c qdb_input96_depth3_x64_8b.yml; python test.py -c qdb_input96_depth3_x64_8b.yml
python train.py -c qdb_input96_depth3_x64_10b.yml; python test.py -c qdb_input96_depth3_x64_10b.yml
python train.py -c qdb_input96_depth3_x64_12b.yml; python test.py -c qdb_input96_depth3_x64_12b.yml
python train.py -c qdb_input96_depth3_x72_8b.yml; python test.py -c qdb_input96_depth3_x72_8b.yml
python train.py -c qdb_input96_depth3_x72_10b.yml; python test.py -c qdb_input96_depth3_x72_10b.yml
python train.py -c qdb_input96_depth3_x72_12b.yml; python test.py -c qdb_input96_depth3_x72_12b.yml