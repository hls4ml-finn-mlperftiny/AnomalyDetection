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


    # hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    # hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    # hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    # hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')



    # backend=convert_config['convert']['Backend'], 
    # clock_period=convert_config['convert']['ClockPeriod'],
    # io_type=convert_config['convert']['IOType'], 
    # interface=convert_config['convert']['Interface'], 
    # if convert_config['convert']['Backend'] == 'VivadoAccelerator':
    #     board = convert_config['convert']['Board']
    #     driver = convert_config['convert']['Driver']
    #     cfg = hls4ml.converters.create_config(
    #         backend=convert_config['convert']['Backend'], 
    #         board=convert_config['convert']['Board'], 
    #         interface=convert_config['convert']['Interface'], 
    #         clock_period=convert_config['convert']['ClockPeriod'],
    #         io_type=convert_config['convert']['IOType'], 
    #         driver=convert_config['convert']['Driver'])
    # else:
    #     part = convert_config['convert']['fpga_part']
    #     cfg = hls4ml.converters.create_config(
    #         backend=convert_config['convert']['Backend'], 
    #         part=convert_config['convert']['fpga_part'],
    #         clock_period=convert_config['convert']['ClockPeriod'],
    #         io_type=convert_config['convert']['IOType'],)

    # cfg['HLSConfig'] = convert_config['hls_config']['HLSConfig']
    # cfg['InputData'] = convert_config['convert']['x_npy_hls_test_bench']
    # cfg['OutputPredictions'] = convert_config['convert']['x_npy_hls_test_bench']
    # cfg['KerasModel'] = model
    # cfg['OutputDir'] = convert_config['convert']['OutputDir']

    # print("-----------------------------------")
    # print_dict(cfg)
    # print("-----------------------------------")

    # # profiling / testing
    # # profiling / testing
    # hls_model = hls4ml.converters.keras_to_hls(cfg)
    # if not os.path.exists(convert_config['convert']['OutputDir']):
    #     os.makedirs(convert_config['convert']['OutputDir'])
    # hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='{}/model_hls4ml.png'.format(convert_config['convert']['OutputDir']))
    # hls_model.compile()
    # plot_roc(model, hls_model, convert_config['convert']['x_npy_plot_roc'], convert_config['convert']['y_npy_plot_roc'], convert_config['convert']['OutputDir'])



    #  # Bitfile time
    # if bool(convert_config['convert']['Build']):
    #     if bool(convert_config['convert']['FIFO_opt']):
    #         from hls4ml.model.profiling import optimize_fifos_depth
    #         hls_model = optimize_fifos_depth(model, output_dir=convert_config['convert']['OutputDir'],
    #                                          clock_period=convert_config['convert']['ClockPeriod'],
    #                                          backend=convert_config['convert']['Backend'],
    #                                          input_data_tb=os.path.join(convert_config['convert']['x_npy_hls_test_bench']),
    #                                          output_data_tb=os.path.join(convert_config['convert']['x_npy_hls_test_bench']),
    #                                          board=convert_config['convert']['Board'], hls_config=convert_config['hls_config']['HLSConfig'])
    #     else:
    #         hls_model.build(reset=False, csim=True, cosim=True, validation=True, synth=True, vsynth=True, export=True)
    #         hls4ml.report.read_vivado_report(convert_config['convert']['OutputDir'])
    #     if convert_config['convert']['Backend'] == 'VivadoAccelerator':
    #         hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)

    #begin test 12/26/2021
    interface = 'm_axi' # 's_axilite', 'm_axi', 'hls_stream'
    axi_width = 8 # 16, 32, 64
    implementation = 'serial' # 'serial', 'dataflow'
    output_dir=f'hls/{convert_config["convert"]["Board"]}_{convert_config["convert"]["acc_name"]}_{interface}_{str(axi_width)}_{implementation}_prj'

    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    # backend_config = hls4ml.converters.create_backend_config(fpga_part=convert_config["convert"]["fpga_part"])
    backend_config = {}
    backend_config['XilinxPart'] = 'xc7z020clg400-1'
    backend_config['ProjectName'] = convert_config["convert"]["acc_name"]
    backend_config['KerasModel'] = model
    backend_config['HLSConfig'] = convert_config["convert"]["HLSConfig"]
    backend_config['OutputDir'] = output_dir
    backend_config['Backend'] = convert_config["convert"]["Backend"]
    backend_config['Interface'] = interface
    backend_config['IOType'] = convert_config["convert"]["IOType"]
    backend_config['AxiWidth'] = str(axi_width)
    backend_config['Implementation'] = implementation
    backend_config['ClockPeriod'] = convert_config["convert"]["ClockPeriod"]
    #print("-----------------------------------")
    # plotting.print_dict(backend_config)
    #print("-----------------------------------")
    hls_model = hls4ml.converters.keras_to_hls(backend_config)
    _ = hls_model.compile()
    plot_roc(model, hls_model, X_npy="test_data/anomaly_detection/downsampled_128_5_to_32_4_skip_method.npy", 
            y_npy="test_data/anomaly_detection/downsampled_128_5_to_32_4_ground_truths_skip_method.npy", output_dir=output_dir)
    hls_model.build(csim=False, synth=True, vsynth=True, export=True)
    # hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="specify yaml config")

    args = parser.parse_args()

    main(args)