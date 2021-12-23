"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################

import setGPU
import os
#import tensorflow as tf
import tensorflow.compat.v2 as tf
# V2 Behavior is necessary to use TF2 APIs before TF2 is default TF version internally.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
import glob
import sys
import time
########################################################################
import logging
########################################################################
# import additional python-library
########################################################################
import numpy
from tqdm import tqdm
# original lib
import common as com
import keras_model
########################################################################
#auto qKeras
########################################################################
import tempfile
tf.enable_v2_behavior()
from tensorflow.keras.optimizers import *

from qkeras.autoqkeras import *
from qkeras import *
from qkeras.utils import model_quantize
from qkeras.qtools import run_qtools
from qkeras.qtools import settings as qtools_settings
import pprint

from tensorflow.keras.utils import to_categorical
#import tensorflow_datasets as tfds
#from tensorflow.keras.datasets import mnist

########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.
        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.
        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.
        name : str
            save png file path.
        return : None
        """
        self.plt.savefig(name)


########################################################################


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.
    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.
    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), 128), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    print("Shape of dataset: {}".format(dataset.shape))
    return dataset


def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files
    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files

########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    args = com.command_line_chk()

    # load parameter.yaml
    param = com.yaml_load(args.config)

    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.h5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)

        if os.path.exists(history_img):
            com.logger.info("model already exists and trained")
            continue


        # generate dataset

        train_data_save_load_directory = "./train_time_data/downsampled_128_5_to_32_4_skip_method.npy"
        # if train_data available, load post processed data in local directory without reprocessing wav files --saves time--
        if os.path.exists(train_data_save_load_directory):
            print("Loading train_data from {}".format(train_data_save_load_directory))

            train_data = numpy.load(train_data_save_load_directory, allow_pickle=True)
        else:
            print("============== DATASET_GENERATOR ==============")
            files = file_list_generator(target_dir)
            train_data = list_to_vector_array(files,
                                              msg="generate train_dataset",
                                              n_mels=param["feature"]["n_mels"],
                                              frames=param["feature"]["frames"],
                                              n_fft=param["feature"]["n_fft"],
                                              hop_length=param["feature"]["hop_length"],
                                              power=param["feature"]["power"])
            #save train_data
            if not os.path.exists('./train_time_data'):
                os.makedirs('./train_time_data')
            numpy.save(train_data_save_load_directory, train_data)
            print("Train data saved to {}".format(train_data_save_load_directory))

        # train model
        print("============== MODEL TRAINING ==============")
        if param["model"]["name"] == 'qkeras_model':
            model = keras_model.get_model(param["model"]["name"], 
                                          2*32,
                                          hiddenDim=param["model"]["hidden_dim"],
                                          encodeDim=param["model"]["encode_dim"],
                                          bits=param["model"]["quantization"]["bits"],
                                          intBits=param["model"]["quantization"]["int_bits"],
                                          reluBits=param["model"]["quantization"]["relu_bits"],
                                          reluIntBits=param["model"]["quantization"]["relu_int_bits"],
                                          lastBits=param["model"]["quantization"]["last_bits"],
                                          lastIntBits=param["model"]["quantization"]["last_int_bits"],
                                          l1reg=param["model"]["l1reg"],
                                          batchNorm=param["model"]["batch_norm"],
                                          halfcode_layers=param["model"]["halfcode_layers"],
                                          fan_in_out=param["model"]["fan_in_out"])
        else:
            model = keras_model.get_model(param["model"]["name"],
                                        2*32,
                                        hiddenDim=param["model"]["hidden_dim"],
                                        encodeDim=param["model"]["encode_dim"],
                                        halfcode_layers=param["model"]["halfcode_layers"],
                                        fan_in_out=param["model"]["fan_in_out"],
                                        batchNorm=param["model"]["batch_norm"],
                                        l1reg=param["model"]["l1reg"],
                                        bits=param["model"]["quantization"]["bits"],
                                        intBits=param["model"]["quantization"]["int_bits"],
                                        reluBits=param["model"]["quantization"]["relu_bits"],
                                        reluIntBits=param["model"]["quantization"]["relu_int_bits"],
                                        lastBits=param["model"]["quantization"]["last_bits"],
                                        lastIntBits=param["model"]["quantization"]["last_int_bits"])
        param["model"]["name"]
        model.summary()
        #get energy profile
        reference_internal = "fp32"
        reference_accumulator = "fp32"

        q = run_qtools.QTools(
          model,
          # energy calculation using a given process
          # "horowitz" refers to 45nm process published at
          # M. Horowitz, "1.1 Computing's energy problem (and what we can do about
          # it), "2014 IEEE International Solid-State Circuits Conference Digest of
          # Technical Papers (ISSCC), San Francisco, CA, 2014, pp. 10-14, 
          # doi: 10.1109/ISSCC.2014.6757323.
          process="horowitz",
          # quantizers for model input
          source_quantizers=[quantized_bits(8, 0, 1)],
          is_inference=False,
          # absolute path (including filename) of the model weights
          # in the future, we will attempt to optimize the power model
          # by using weight information, although it can be used to further
          # optimize QBatchNormalization.
          weights_path=None,
          # keras_quantizer to quantize weight/bias in un-quantized keras layers
          keras_quantizer=reference_internal,
          # keras_quantizer to quantize MAC in un-quantized keras layers
          keras_accumulator=reference_accumulator,
          # whether calculate baseline energy
          for_reference=True)

        # caculate energy of the derived data type map.
        energy_dict = q.pe(
        # whether to store parameters in dram, sram, or fixed
        weights_on_memory="sram",
        # store activations in dram or sram
        activations_on_memory="sram",
        # minimum sram size in number of bits. Let's assume a 16MB SRAM.
        min_sram_size=8*16*1024*1024,
        # whether load data from dram to sram (consider sram as a cache
        # for dram. If false, we will assume data will be already in SRAM
        rd_wr_on_io=False)

        # get stats of energy distribution in each layer
        energy_profile = q.extract_energy_profile(
        qtools_settings.cfg.include_energy, energy_dict)
        # extract sum of energy of each layer according to the rule specified in
        # qtools_settings.cfg.include_energy
        total_energy = q.extract_energy_sum(
        qtools_settings.cfg.include_energy, energy_dict)

        pprint.pprint(energy_profile)
        print()
        print("Total energy: {:.2f} uJ".format(total_energy / 1000000.0))

        custom_objects = {}
        cur_strategy = tf.distribute.get_strategy()

        ########################################################################
        #autoQK Config
        ########################################################################
        quantization_config = {
                "kernel": {
                        "binary": 1,
                        "stochastic_binary": 1,
                        "ternary": 2,
                        "stochastic_ternary": 2,
                        "quantized_bits(2,1,1,alpha=1.0)": 2,
                        "quantized_bits(4,0,1,alpha=1.0)": 4,
                        "quantized_bits(8,0,1,alpha=1.0)": 8,
                        "quantized_po2(4,1)": 4
                },
                "bias": {
                        "quantized_bits(4,0,1)": 4,
                        "quantized_bits(8,3,1)": 8,
                        "quantized_po2(4,8)": 4
                },
                "activation": {
                        "binary": 1,
                        "ternary": 2,
                        "quantized_relu_po2(4,4)": 4,
                        "quantized_relu(3,1)": 3,
                        "quantized_relu(4,2)": 4,
                        "quantized_relu(8,2)": 8,
                        "quantized_relu(8,4)": 8,
                        "quantized_relu(16,8)": 16
                },
                "linear": {
                        "binary": 1,
                        "ternary": 2,
                        "quantized_bits(4,1)": 4,
                        "quantized_bits(8,2)": 8,
                        "quantized_bits(16,10)": 16
                }
        }

        limit = {
            "Dense": [8, 8, 4],
            "Conv2D": [4, 8, 4],
            "DepthwiseConv2D": [4, 8, 4],
            "Activation": [['quantized_relu', 'quantized_tanh', 'binary_tanh', 'ternary_tanh', 'quantized_bits', 'binary', 'ternary'], 4],
#             "BatchNormalization": [] Commented out to avoid qBatchnorm layers which are not supported by hls4ml
            
        }
        goal = {
            "type": "energy",
            "params": {
                "delta_p": 8.0,
                "delta_n": 2.0,
                "rate": 4.0,
                "stress": 1.0,
                "process": "horowitz",
                "parameters_on_memory": ["sram", "sram"],
                "activations_on_memory": ["sram", "sram"],
                "rd_wr_on_io": [False, False],
                "min_sram_size": [0, 0],
                "source_quantizers": ["int8"],
                "reference_internal": "int8",
                "reference_accumulator": "int32"
                }
        }

        run_config = {
          "output_dir": tempfile.mkdtemp(),
          "goal": goal,
          "quantization_config": quantization_config,
          "learning_rate_optimizer": False,
          "transfer_weights": False,
          "mode": "bayesian",
          "seed": 42,
          "limit": limit,
          "tune_filters": "layer",
          "tune_filters_exceptions": "^dense",
          "distribution_strategy": cur_strategy,
          # first layer is input, layer two layers are softmax and flatten
          "layer_indexes": range(1, len(model.layers) - 1),
          "max_trials": 10
        }
        print("quantizing layers:", [model.layers[i].name for i in run_config["layer_indexes"]])

        from tensorflow.keras.callbacks import EarlyStopping,History,ModelCheckpoint,ReduceLROnPlateau, TensorBoard

        modelbestcheck = ModelCheckpoint(model_file_path,
                                         monitor='val_loss',
                                         verbose=2,
                                         save_best_only=True)
        stopping = EarlyStopping(monitor='val_loss',
                                 patience = 10 if param["pruning"]["constant"] == True else 10 if param["pruning"]["decay"] == True else 15, verbose=1, mode='min')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1,
                                      mode='min', verbose=1, epsilon=0.001,
                                      cooldown=4, min_lr=1e-5)

        callbacks=[
            modelbestcheck,
            stopping,
            reduce_lr,
        ]
        autoqCallbacks = [
                stopping,
                reduce_lr,
        ]

        model.compile(**param["fit"]["compile"])

        print("Shape of training data element is: {}".format(numpy.shape(train_data[0])))

        print("AutoQK")
        autoqk = AutoQKeras(model, metrics=["acc"], custom_objects=custom_objects, **run_config)
        history = autoqk.fit(train_data,
                             train_data,
                             epochs=param["fit"]["autoqEpochs"],
                             batch_size=param["fit"]["batch_size"],
                             shuffle=param["fit"]["shuffle"],
                             validation_split=param["fit"]["validation_split"],
                             verbose=param["fit"]["verbose"],
                             callbacks=autoqCallbacks)
                            
        qmodel = autoqk.get_best_model()

        with cur_strategy.scope():
          qmodel.compile(**param["fit"]["compile"])
          qmodel.fit(train_data,
                            train_data,
                            epochs=param["fit"]["epochs"],
                            batch_size=param["fit"]["batch_size"],
                            shuffle=param["fit"]["shuffle"],
                            validation_split=param["fit"]["validation_split"],
                            verbose=param["fit"]["verbose"],
                            callbacks=autoqCallbacks)
        qmodel.save(model_file_path)

        com.logger.info("save_model -> {}".format(model_file_path))
        qmodel.summary()
        print("============== END TRAINING ==============")
