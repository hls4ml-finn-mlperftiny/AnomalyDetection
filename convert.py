import tensorflow as tf
import yaml
import argparse
import os
import numpy as np
from plot_roc import plot_roc

import hls4ml

import matplotlib.pyplot as plt

def is_tool(name):
    from distutils.spawn import find_executable
    return find_executable(name) is not None

def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))

def yaml_load(config):
        with open(config, 'r') as stream:
            param = yaml.safe_load(stream)
        return param
  
def load_model(file_path):
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)  
    return tf.keras.models.load_model(file_path, custom_objects=co)


def main(args):

    convert_config = yaml_load(args.config)

    os.environ['PATH'] = convert_config['convert']['vivado_path'] + os.environ['PATH']
    print('-----------------------------------')
    if not is_tool('vivado_hls'):
        print('Xilinx Vivado HLS is NOT in the PATH')
    else:
        print('Xilinx Vivado HLS is in the PATH')
    print('-----------------------------------')

    # test bench data
    # X_npy = np.load(convert_config['convert']['x_npy_hls_test_bench'], allow_pickle=True)
    # y_npy =np.load(convert_config['convert']['y_npy_hls_test_bench'], allow_pickle=True)

    model = load_model(convert_config['convert']['model_file'])
    model.summary()


    import hls4ml



    # hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    # hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    # hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    # hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')




 
interface = 'm_axi' # 's_axilite', 'm_axi', 'hls_stream'
axi_width = 16 # 16, 32, 64
implementation = 'serial' # 'serial', 'dataflow'
output_dir='hls/autoretest/' + board_name + '_' + interface + '_' + str(axi_width) + '_' + implementation + '_prj' 
hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       output_dir=output_dir,
                                                       project_name='anomaly_detector',
                                                       fpga_part=fpga_part,
                                                       clock_period=10,
                                                       io_type='io_parallel',
                                                       hls_config=hls_config,
                                                       backend='Pynq',)


_ = hls_model.compile()
hls_model.build(csim=False,synth=True,vsynth=True,export=True)

hls4ml.report.read_vivado_report(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="baseline.yml", help="specify yaml config")

    args = parser.parse_args()

    main(args)