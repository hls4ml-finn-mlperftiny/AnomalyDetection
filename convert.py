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
    X_npy = np.load(convert_config['convert']['x_npy_hls_test_bench'], allow_pickle=True)
    y_npy =np.load(convert_config['convert']['y_npy_hls_test_bench'], allow_pickle=True)

    model = load_model(convert_config['convert']['model_file'])
    model.summary()


    hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'
    hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    hls_config['Model']['ReuseFactor'] = 16384
    hls_config['Model']['Strategy'] = 'Resource'
    hls_config['Model']['Precision'] = 'ap_fixed<32,16>'
    hls_config['LayerName']['input_1']['Precision'] = 'ap_fixed<8,8>'
    for layer in hls_config['LayerName'].keys():
        hls_config['LayerName'][layer]['Trace'] = True
        hls_config['LayerName'][layer]['ReuseFactor'] = 16384 #large number, hls4ml will clip to #inputs x layer_width
        hls_config['LayerName'][layer]['accum_t'] = 'ap_fixed<32,16>'



    backend=convert_config['convert']['Backend'], 
    clock_period=convert_config['convert']['ClockPeriod'],
    io_type=convert_config['convert']['IOType'], 
    interface=convert_config['convert']['Interface'], 
    if convert_config['convert']['Backend'] == 'VivadoAccelerator':
        board = convert_config['convert']['Board']
        driver = convert_config['convert']['Driver']
        cfg = hls4ml.converters.create_config(
            backend=convert_config['convert']['Backend'], 
            board=convert_config['convert']['Board'], 
            interface=convert_config['convert']['Interface'], 
            clock_period=convert_config['convert']['ClockPeriod'],
            io_type=convert_config['convert']['IOType'], 
            driver=convert_config['convert']['Driver'])
    else:
        part = convert_config['convert']['fpga_part']
        cfg = hls4ml.converters.create_config(
            backend=convert_config['convert']['Backend'], 
            part=convert_config['convert']['fpga_part'],
            clock_period=convert_config['convert']['ClockPeriod'],
            io_type=convert_config['convert']['IOType'],)

    # cfg['HLSConfig'] = convert_config['hls_config']['HLSConfig']
    cfg['HLSConfig'] = hls_config
    cfg['InputData'] = convert_config['convert']['x_npy_hls_test_bench']
    cfg['OutputPredictions'] = convert_config['convert']['x_npy_hls_test_bench']
    cfg['KerasModel'] = model
    cfg['OutputDir'] = convert_config['convert']['OutputDir']

    print("-----------------------------------")
    print_dict(cfg)
    print("-----------------------------------")

    hls_model = hls4ml.converters.keras_to_hls(cfg)
    if not os.path.exists(convert_config['convert']['OutputDir']):
        os.makedirs(convert_config['convert']['OutputDir'])
    hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='{}/model_hls4ml.png'.format(convert_config['convert']['OutputDir']))
    hls_model.compile()

    # profiling / testing
    # profiling data
    X_prof = np.load(convert_config['convert']['x_npy_plot_roc'], allow_pickle=True)
    hls4ml_pred, hls4ml_trace = hls_model.trace(np.ascontiguousarray(X_prof[0][0][0]))
    # Run tracing on a portion of the test set for the Keras model (floating-point precision)
    keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_prof[0][0])
    for key in hls4ml_trace:
        plt.figure()
        plt.scatter(keras_trace[key][0], hls4ml_trace[key][0], color='black')
        plt.plot(np.linspace(np.min(keras_trace[key][0]),np.max(keras_trace[key][0]), 10), 
                np.linspace(np.min(keras_trace[key][0]),np.max(keras_trace[key][0]), 10), label='keras_range')
        plt.plot(np.linspace(np.min(hls4ml_trace[key][0]),np.max(hls4ml_trace[key][0]), 10), 
                np.linspace(np.min(hls4ml_trace[key][0]),np.max(hls4ml_trace[key][0]), 10), label='hls4ml_range')
        plt.title(key)
        plt.xlabel('keras output')
        plt.ylabel('hls4ml output')
        plt.legend()

        plt.savefig(convert_config['convert']['OutputDir']+'hls4ml_v_keras_trace/{}'.format(key))
        print('profiled layer {}'.format(key))
    plot_roc(model, hls_model, convert_config['convert']['x_npy_plot_roc'], convert_config['convert']['y_npy_plot_roc'], convert_config['convert']['OutputDir'])




     # Bitfile time
    if bool(convert_config['convert']['Build']):
        if bool(convert_config['convert']['FIFO_opt']):
            from hls4ml.model.profiling import optimize_fifos_depth
            hls_model = optimize_fifos_depth(model, output_dir=convert_config['convert']['OutputDir'],
                                             clock_period=convert_config['convert']['ClockPeriod'],
                                             backend=convert_config['convert']['Backend'],
                                             input_data_tb=os.path.join(convert_config['convert']['x_npy_hls_test_bench']),
                                             output_data_tb=os.path.join(convert_config['convert']['x_npy_hls_test_bench']),
                                             board=convert_config['convert']['Board'], hls_config=convert_config['hls_config']['HLSConfig'])
        else:
            hls_model.build(reset=False, csim=True, cosim=True, validation=True, synth=True, vsynth=True, export=True)
            hls4ml.report.read_vivado_report(convert_config['convert']['OutputDir'])
        if convert_config['convert']['Backend'] == 'VivadoAccelerator':
            hls4ml.templates.VivadoAcceleratorBackend.make_bitfile(hls_model)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="specify yaml config")

    args = parser.parse_args()

    main(args)