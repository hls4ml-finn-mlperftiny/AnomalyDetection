#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 128
#define N_LAYER_2 64
#define N_LAYER_6 8
#define N_LAYER_10 64
#define N_LAYER_14 128

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<8,8> input_1_default_t;
typedef nnet::array<ap_fixed<8,8>, 128*1> input_t;
typedef ap_fixed<32,16> model_default_t;
typedef nnet::array<ap_fixed<32,16>, 64*1> layer2_t;
typedef ap_fixed<10,1> weight2_t;
typedef ap_fixed<10,1> bias2_t;
typedef nnet::array<ap_fixed<32,16>, 64*1> layer4_t;
typedef ap_fixed<16,6> batch_normalization_scale_t;
typedef ap_fixed<16,6> batch_normalization_bias_t;
typedef nnet::array<ap_fixed<8,4>, 64*1> layer5_t;
typedef nnet::array<ap_fixed<32,16>, 8*1> layer6_t;
typedef ap_fixed<10,1> weight6_t;
typedef ap_fixed<10,1> bias6_t;
typedef nnet::array<ap_fixed<32,16>, 8*1> layer8_t;
typedef ap_fixed<16,6> batch_normalization_1_scale_t;
typedef ap_fixed<16,6> batch_normalization_1_bias_t;
typedef nnet::array<ap_fixed<8,4>, 8*1> layer9_t;
typedef nnet::array<ap_fixed<32,16>, 64*1> layer10_t;
typedef ap_fixed<10,1> weight10_t;
typedef ap_fixed<10,1> bias10_t;
typedef nnet::array<ap_fixed<32,16>, 64*1> layer12_t;
typedef ap_fixed<16,6> batch_normalization_2_scale_t;
typedef ap_fixed<16,6> batch_normalization_2_bias_t;
typedef nnet::array<ap_fixed<8,4>, 64*1> layer13_t;
typedef nnet::array<ap_fixed<32,16>, 128*1> layer14_t;
typedef ap_int<10> weight14_t;
typedef ap_int<10> bias14_t;

#endif
