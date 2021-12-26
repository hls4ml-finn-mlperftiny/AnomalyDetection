#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/s4.h"
#include "weights/b4.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/s8.h"
#include "weights/b8.h"
#include "weights/w10.h"
#include "weights/b10.h"
#include "weights/s12.h"
#include "weights/b12.h"
#include "weights/w14.h"
#include "weights/b14.h"

//hls-fpga-machine-learning insert layer-config
// q_dense
struct config2 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_out = N_LAYER_2;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 4096;
    static const unsigned n_zeros = 41;
    static const unsigned n_nonzeros = 8151;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<32,16> accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// batch_normalization
struct config4 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4096;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_bias_t bias_t;
    typedef batch_normalization_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_activation
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4096;
    typedef ap_fixed<18,8> table_t;
};

// q_dense_1
struct config6 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_out = N_LAYER_6;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 512;
    static const unsigned n_zeros = 7;
    static const unsigned n_nonzeros = 505;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<32,16> accum_t;
    typedef bias6_t bias_t;
    typedef weight6_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// batch_normalization_1
struct config8 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4096;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_1_bias_t bias_t;
    typedef batch_normalization_1_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_activation_1
struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4096;
    typedef ap_fixed<18,8> table_t;
};

// q_dense_2
struct config10 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned n_out = N_LAYER_10;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 512;
    static const unsigned n_zeros = 6;
    static const unsigned n_nonzeros = 506;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<32,16> accum_t;
    typedef bias10_t bias_t;
    typedef weight10_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// batch_normalization_2
struct config12 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_10;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4096;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_2_bias_t bias_t;
    typedef batch_normalization_2_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_activation_2
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 4096;
    typedef ap_fixed<18,8> table_t;
};

// q_dense_3
struct config14 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_10;
    static const unsigned n_out = N_LAYER_14;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 4096;
    static const unsigned n_zeros = 469;
    static const unsigned n_nonzeros = 7723;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<32,16> accum_t;
    typedef bias14_t bias_t;
    typedef weight14_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};


#endif
