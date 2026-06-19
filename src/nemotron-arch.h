#pragma once

#include "ggml.h"

#include <map>

#define NEMOTRON_N_AUDIO_LAYER       24
#define NEMOTRON_N_TENSORS_PER_LAYER 26
#define NEMOTRON_N_MODEL_TENSORS     657

enum nemotron_tensor {
    // Preprocessor
    NEMOTRON_TENSOR_PREPROCESSOR_FB,
    NEMOTRON_TENSOR_PREPROCESSOR_WINDOW,

    // Encoder pre_encode
    NEMOTRON_TENSOR_ENC_PRE_OUT_WEIGHT,
    NEMOTRON_TENSOR_ENC_PRE_OUT_BIAS,
    NEMOTRON_TENSOR_ENC_PRE_CONV_0_WEIGHT,
    NEMOTRON_TENSOR_ENC_PRE_CONV_0_BIAS,
    NEMOTRON_TENSOR_ENC_PRE_CONV_2_WEIGHT,
    NEMOTRON_TENSOR_ENC_PRE_CONV_2_BIAS,
    NEMOTRON_TENSOR_ENC_PRE_CONV_3_WEIGHT,
    NEMOTRON_TENSOR_ENC_PRE_CONV_3_BIAS,
    NEMOTRON_TENSOR_ENC_PRE_CONV_5_WEIGHT,
    NEMOTRON_TENSOR_ENC_PRE_CONV_5_BIAS,
    NEMOTRON_TENSOR_ENC_PRE_CONV_6_WEIGHT,
    NEMOTRON_TENSOR_ENC_PRE_CONV_6_BIAS,

    // Prompt kernel
    NEMOTRON_TENSOR_PROMPT_KERNEL_0_WEIGHT,
    NEMOTRON_TENSOR_PROMPT_KERNEL_0_BIAS,
    NEMOTRON_TENSOR_PROMPT_KERNEL_2_WEIGHT,
    NEMOTRON_TENSOR_PROMPT_KERNEL_2_BIAS,

    // Encoder layers (per-layer)
    NEMOTRON_TENSOR_ENC_NORM_FF1_WEIGHT,
    NEMOTRON_TENSOR_ENC_NORM_FF1_BIAS,
    NEMOTRON_TENSOR_ENC_FF1_LINEAR1_WEIGHT,
    NEMOTRON_TENSOR_ENC_FF1_LINEAR2_WEIGHT,
    NEMOTRON_TENSOR_ENC_NORM_ATTN_WEIGHT,
    NEMOTRON_TENSOR_ENC_NORM_ATTN_BIAS,
    NEMOTRON_TENSOR_ENC_ATTN_Q_WEIGHT,
    NEMOTRON_TENSOR_ENC_ATTN_K_WEIGHT,
    NEMOTRON_TENSOR_ENC_ATTN_V_WEIGHT,
    NEMOTRON_TENSOR_ENC_ATTN_OUT_WEIGHT,
    NEMOTRON_TENSOR_ENC_ATTN_POS_WEIGHT,
    NEMOTRON_TENSOR_ENC_ATTN_POS_BIAS_U,
    NEMOTRON_TENSOR_ENC_ATTN_POS_BIAS_V,
    NEMOTRON_TENSOR_ENC_NORM_CONV_WEIGHT,
    NEMOTRON_TENSOR_ENC_NORM_CONV_BIAS,
    NEMOTRON_TENSOR_ENC_CONV_PW1_WEIGHT,
    NEMOTRON_TENSOR_ENC_CONV_DW_WEIGHT,
    NEMOTRON_TENSOR_ENC_CONV_BN_WEIGHT,
    NEMOTRON_TENSOR_ENC_CONV_BN_BIAS,
    NEMOTRON_TENSOR_ENC_CONV_PW2_WEIGHT,
    NEMOTRON_TENSOR_ENC_NORM_FF2_WEIGHT,
    NEMOTRON_TENSOR_ENC_NORM_FF2_BIAS,
    NEMOTRON_TENSOR_ENC_FF2_LINEAR1_WEIGHT,
    NEMOTRON_TENSOR_ENC_FF2_LINEAR2_WEIGHT,
    NEMOTRON_TENSOR_ENC_NORM_OUT_WEIGHT,
    NEMOTRON_TENSOR_ENC_NORM_OUT_BIAS,

    // Prediction network
    NEMOTRON_TENSOR_PRED_EMBED_WEIGHT,
    NEMOTRON_TENSOR_PRED_LSTM_WEIGHT_IH,
    NEMOTRON_TENSOR_PRED_LSTM_WEIGHT_HH,
    NEMOTRON_TENSOR_PRED_LSTM_BIAS_IH,
    NEMOTRON_TENSOR_PRED_LSTM_BIAS_HH,

    // Joint network
    NEMOTRON_TENSOR_JOINT_ENC_WEIGHT,
    NEMOTRON_TENSOR_JOINT_ENC_BIAS,
    NEMOTRON_TENSOR_JOINT_PRED_WEIGHT,
    NEMOTRON_TENSOR_JOINT_PRED_BIAS,
    NEMOTRON_TENSOR_JOINT_NET_WEIGHT,
    NEMOTRON_TENSOR_JOINT_NET_BIAS,
};

static const std::map<nemotron_tensor, const char *> NEMOTRON_TENSOR_NAMES = {
    // Preprocessor
    {NEMOTRON_TENSOR_PREPROCESSOR_FB,             "preprocessor.featurizer.fb"},
    {NEMOTRON_TENSOR_PREPROCESSOR_WINDOW,         "preprocessor.featurizer.window"},

    // Encoder pre_encode
    {NEMOTRON_TENSOR_ENC_PRE_OUT_WEIGHT,          "encoder.pre_encode.out.weight"},
    {NEMOTRON_TENSOR_ENC_PRE_OUT_BIAS,            "encoder.pre_encode.out.bias"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_0_WEIGHT,       "encoder.pre_encode.conv.0.weight"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_0_BIAS,         "encoder.pre_encode.conv.0.bias"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_2_WEIGHT,       "encoder.pre_encode.conv.2.weight"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_2_BIAS,         "encoder.pre_encode.conv.2.bias"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_3_WEIGHT,       "encoder.pre_encode.conv.3.weight"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_3_BIAS,         "encoder.pre_encode.conv.3.bias"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_5_WEIGHT,       "encoder.pre_encode.conv.5.weight"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_5_BIAS,         "encoder.pre_encode.conv.5.bias"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_6_WEIGHT,       "encoder.pre_encode.conv.6.weight"},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_6_BIAS,         "encoder.pre_encode.conv.6.bias"},

    // Prompt kernel
    {NEMOTRON_TENSOR_PROMPT_KERNEL_0_WEIGHT,      "prompt_kernel.0.weight"},
    {NEMOTRON_TENSOR_PROMPT_KERNEL_0_BIAS,        "prompt_kernel.0.bias"},
    {NEMOTRON_TENSOR_PROMPT_KERNEL_2_WEIGHT,      "prompt_kernel.2.weight"},
    {NEMOTRON_TENSOR_PROMPT_KERNEL_2_BIAS,        "prompt_kernel.2.bias"},

    // Encoder layers (use %d for layer number)
    {NEMOTRON_TENSOR_ENC_NORM_FF1_WEIGHT,         "encoder.layers.%d.norm_feed_forward1.weight"},
    {NEMOTRON_TENSOR_ENC_NORM_FF1_BIAS,           "encoder.layers.%d.norm_feed_forward1.bias"},
    {NEMOTRON_TENSOR_ENC_FF1_LINEAR1_WEIGHT,      "encoder.layers.%d.feed_forward1.linear1.weight"},
    {NEMOTRON_TENSOR_ENC_FF1_LINEAR2_WEIGHT,      "encoder.layers.%d.feed_forward1.linear2.weight"},
    {NEMOTRON_TENSOR_ENC_NORM_ATTN_WEIGHT,        "encoder.layers.%d.norm_self_att.weight"},
    {NEMOTRON_TENSOR_ENC_NORM_ATTN_BIAS,          "encoder.layers.%d.norm_self_att.bias"},
    {NEMOTRON_TENSOR_ENC_ATTN_Q_WEIGHT,           "encoder.layers.%d.self_attn.linear_q.weight"},
    {NEMOTRON_TENSOR_ENC_ATTN_K_WEIGHT,           "encoder.layers.%d.self_attn.linear_k.weight"},
    {NEMOTRON_TENSOR_ENC_ATTN_V_WEIGHT,           "encoder.layers.%d.self_attn.linear_v.weight"},
    {NEMOTRON_TENSOR_ENC_ATTN_OUT_WEIGHT,         "encoder.layers.%d.self_attn.linear_out.weight"},
    {NEMOTRON_TENSOR_ENC_ATTN_POS_WEIGHT,         "encoder.layers.%d.self_attn.linear_pos.weight"},
    {NEMOTRON_TENSOR_ENC_ATTN_POS_BIAS_U,         "encoder.layers.%d.self_attn.pos_bias_u"},
    {NEMOTRON_TENSOR_ENC_ATTN_POS_BIAS_V,         "encoder.layers.%d.self_attn.pos_bias_v"},
    {NEMOTRON_TENSOR_ENC_NORM_CONV_WEIGHT,        "encoder.layers.%d.norm_conv.weight"},
    {NEMOTRON_TENSOR_ENC_NORM_CONV_BIAS,          "encoder.layers.%d.norm_conv.bias"},
    {NEMOTRON_TENSOR_ENC_CONV_PW1_WEIGHT,         "encoder.layers.%d.conv.pointwise_conv1.weight"},
    {NEMOTRON_TENSOR_ENC_CONV_DW_WEIGHT,          "encoder.layers.%d.conv.depthwise_conv.weight"},
    {NEMOTRON_TENSOR_ENC_CONV_BN_WEIGHT,          "encoder.layers.%d.conv.batch_norm.weight"},
    {NEMOTRON_TENSOR_ENC_CONV_BN_BIAS,            "encoder.layers.%d.conv.batch_norm.bias"},
    {NEMOTRON_TENSOR_ENC_CONV_PW2_WEIGHT,         "encoder.layers.%d.conv.pointwise_conv2.weight"},
    {NEMOTRON_TENSOR_ENC_NORM_FF2_WEIGHT,         "encoder.layers.%d.norm_feed_forward2.weight"},
    {NEMOTRON_TENSOR_ENC_NORM_FF2_BIAS,           "encoder.layers.%d.norm_feed_forward2.bias"},
    {NEMOTRON_TENSOR_ENC_FF2_LINEAR1_WEIGHT,      "encoder.layers.%d.feed_forward2.linear1.weight"},
    {NEMOTRON_TENSOR_ENC_FF2_LINEAR2_WEIGHT,      "encoder.layers.%d.feed_forward2.linear2.weight"},
    {NEMOTRON_TENSOR_ENC_NORM_OUT_WEIGHT,         "encoder.layers.%d.norm_out.weight"},
    {NEMOTRON_TENSOR_ENC_NORM_OUT_BIAS,           "encoder.layers.%d.norm_out.bias"},

    // Prediction network
    {NEMOTRON_TENSOR_PRED_EMBED_WEIGHT,           "decoder.prediction.embed.weight"},
    {NEMOTRON_TENSOR_PRED_LSTM_WEIGHT_IH,         "decoder.prediction.dec_rnn.lstm.weight_ih_l%d"},
    {NEMOTRON_TENSOR_PRED_LSTM_WEIGHT_HH,         "decoder.prediction.dec_rnn.lstm.weight_hh_l%d"},
    {NEMOTRON_TENSOR_PRED_LSTM_BIAS_IH,           "decoder.prediction.dec_rnn.lstm.bias_ih_l%d"},
    {NEMOTRON_TENSOR_PRED_LSTM_BIAS_HH,           "decoder.prediction.dec_rnn.lstm.bias_hh_l%d"},

    // Joint network
    {NEMOTRON_TENSOR_JOINT_ENC_WEIGHT,            "joint.enc.weight"},
    {NEMOTRON_TENSOR_JOINT_ENC_BIAS,              "joint.enc.bias"},
    {NEMOTRON_TENSOR_JOINT_PRED_WEIGHT,           "joint.pred.weight"},
    {NEMOTRON_TENSOR_JOINT_PRED_BIAS,             "joint.pred.bias"},
    {NEMOTRON_TENSOR_JOINT_NET_WEIGHT,            "joint.joint_net.2.weight"},
    {NEMOTRON_TENSOR_JOINT_NET_BIAS,              "joint.joint_net.2.bias"},
};

static const std::map<nemotron_tensor, ggml_op> NEMOTRON_TENSOR_INFO = {
    // Preprocessor
    {NEMOTRON_TENSOR_PREPROCESSOR_FB,             GGML_OP_NONE},
    {NEMOTRON_TENSOR_PREPROCESSOR_WINDOW,         GGML_OP_NONE},

    // Encoder pre_encode
    {NEMOTRON_TENSOR_ENC_PRE_OUT_WEIGHT,          GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_PRE_OUT_BIAS,            GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_0_WEIGHT,       GGML_OP_IM2COL},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_0_BIAS,         GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_2_WEIGHT,       GGML_OP_IM2COL},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_2_BIAS,         GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_3_WEIGHT,       GGML_OP_IM2COL},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_3_BIAS,         GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_5_WEIGHT,       GGML_OP_IM2COL},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_5_BIAS,         GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_6_WEIGHT,       GGML_OP_IM2COL},
    {NEMOTRON_TENSOR_ENC_PRE_CONV_6_BIAS,         GGML_OP_ADD},

    // Prompt kernel
    {NEMOTRON_TENSOR_PROMPT_KERNEL_0_WEIGHT,      GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_PROMPT_KERNEL_0_BIAS,        GGML_OP_ADD},
    {NEMOTRON_TENSOR_PROMPT_KERNEL_2_WEIGHT,      GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_PROMPT_KERNEL_2_BIAS,        GGML_OP_ADD},

    // Encoder layers
    {NEMOTRON_TENSOR_ENC_NORM_FF1_WEIGHT,         GGML_OP_MUL},
    {NEMOTRON_TENSOR_ENC_NORM_FF1_BIAS,           GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_FF1_LINEAR1_WEIGHT,      GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_FF1_LINEAR2_WEIGHT,      GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_NORM_ATTN_WEIGHT,        GGML_OP_MUL},
    {NEMOTRON_TENSOR_ENC_NORM_ATTN_BIAS,          GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_ATTN_Q_WEIGHT,           GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_ATTN_K_WEIGHT,           GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_ATTN_V_WEIGHT,           GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_ATTN_OUT_WEIGHT,         GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_ATTN_POS_WEIGHT,         GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_ATTN_POS_BIAS_U,         GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_ATTN_POS_BIAS_V,         GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_NORM_CONV_WEIGHT,        GGML_OP_MUL},
    {NEMOTRON_TENSOR_ENC_NORM_CONV_BIAS,          GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_CONV_PW1_WEIGHT,         GGML_OP_IM2COL},
    {NEMOTRON_TENSOR_ENC_CONV_DW_WEIGHT,          GGML_OP_IM2COL},
    {NEMOTRON_TENSOR_ENC_CONV_BN_WEIGHT,          GGML_OP_MUL},
    {NEMOTRON_TENSOR_ENC_CONV_BN_BIAS,            GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_CONV_PW2_WEIGHT,         GGML_OP_IM2COL},
    {NEMOTRON_TENSOR_ENC_NORM_FF2_WEIGHT,         GGML_OP_MUL},
    {NEMOTRON_TENSOR_ENC_NORM_FF2_BIAS,           GGML_OP_ADD},
    {NEMOTRON_TENSOR_ENC_FF2_LINEAR1_WEIGHT,      GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_FF2_LINEAR2_WEIGHT,      GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_ENC_NORM_OUT_WEIGHT,         GGML_OP_MUL},
    {NEMOTRON_TENSOR_ENC_NORM_OUT_BIAS,           GGML_OP_ADD},

    // Prediction network
    {NEMOTRON_TENSOR_PRED_EMBED_WEIGHT,           GGML_OP_GET_ROWS},
    {NEMOTRON_TENSOR_PRED_LSTM_WEIGHT_IH,         GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_PRED_LSTM_WEIGHT_HH,         GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_PRED_LSTM_BIAS_IH,           GGML_OP_ADD},
    {NEMOTRON_TENSOR_PRED_LSTM_BIAS_HH,           GGML_OP_ADD},

    // Joint network
    {NEMOTRON_TENSOR_JOINT_ENC_WEIGHT,            GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_JOINT_ENC_BIAS,              GGML_OP_ADD},
    {NEMOTRON_TENSOR_JOINT_PRED_WEIGHT,           GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_JOINT_PRED_BIAS,             GGML_OP_ADD},
    {NEMOTRON_TENSOR_JOINT_NET_WEIGHT,            GGML_OP_MUL_MAT},
    {NEMOTRON_TENSOR_JOINT_NET_BIAS,              GGML_OP_ADD},
};
