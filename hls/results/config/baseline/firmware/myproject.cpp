//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_1,
    hls::stream<layer14_t> &layer14_out,
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,layer14_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1;
    const_size_out_1 = N_LAYER_14;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 8192>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 64>(b2, "b2.txt");
        nnet::load_weights_from_txt<batch_normalization_scale_t, 64>(s4, "s4.txt");
        nnet::load_weights_from_txt<batch_normalization_bias_t, 64>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight6_t, 512>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 8>(b6, "b6.txt");
        nnet::load_weights_from_txt<batch_normalization_1_scale_t, 8>(s8, "s8.txt");
        nnet::load_weights_from_txt<batch_normalization_1_bias_t, 8>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight10_t, 512>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 64>(b10, "b10.txt");
        nnet::load_weights_from_txt<batch_normalization_2_scale_t, 64>(s12, "s12.txt");
        nnet::load_weights_from_txt<batch_normalization_2_bias_t, 64>(b12, "b12.txt");
        nnet::load_weights_from_txt<weight14_t, 8192>(w14, "w14.txt");
        nnet::load_weights_from_txt<bias14_t, 128>(b14, "b14.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=1
    nnet::dense<input_t, layer2_t, config2>(input_1, layer2_out, w2, b2); // q_dense

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=1
    nnet::normalize<layer2_t, layer4_t, config4>(layer2_out, layer4_out, s4, b4); // batch_normalization

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=1
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // q_activation

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=1
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // q_dense_1

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=1
    nnet::normalize<layer6_t, layer8_t, config8>(layer6_out, layer8_out, s8, b8); // batch_normalization_1

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=1
    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // q_activation_1

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=1
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // q_dense_2

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=1
    nnet::normalize<layer10_t, layer12_t, config12>(layer10_out, layer12_out, s12, b12); // batch_normalization_2

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=1
    nnet::relu<layer12_t, layer13_t, relu_config13>(layer12_out, layer13_out); // q_activation_2

    nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14); // q_dense_3

}
