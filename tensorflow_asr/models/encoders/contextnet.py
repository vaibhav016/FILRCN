# Copyright 2020 Huy Le Nguyen (@usimarit)
# Copyright 2021 Vaibhav Singh (@vaibhav016)
# Copyright 2021 Dr Vinayak Abrol
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Ref: https://github.com/iankur/ContextNet """

from typing import List
import tensorflow as tf
from ...utils import math_util

L2 = tf.keras.regularizers.l2(1e-6)


def get_activation(activation: str = "silu"):
    activation = activation.lower()
    if activation in ["silu", "swish"]:
        return tf.nn.swish
    elif activation == "relu":
        return tf.nn.relu
    elif activation == "linear":
        return tf.keras.activations.linear
    elif activation == "elu":
        return tf.nn.elu
    else:
        raise ValueError("activation must be either 'silu', 'swish', 'relu' or 'linear' or 'elu' ")


def get_conv_module(lrcn, kernel_size: int = 3, strides=1, filters: int = 256, activation: str = 'silu', kernel_initializer='glorot_uniform',
                    kernel_regularizer=None,
                    bias_regularizer=None, name=None):
    if lrcn:
        return ConvModuleLR(
            kernel_size=kernel_size, strides=strides,
            filters=filters, activation=activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            name=name
        )
    else:
        return ConvModule(
            kernel_size=kernel_size, strides=strides,
            filters=filters, activation=activation, kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            name=name
        )


class Reshape(tf.keras.layers.Layer):
    def call(self, inputs): return math_util.merge_two_last_dims(inputs)


class ConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 filters: int = 256,
                 activation: str = "silu",
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(ConvModule, self).__init__(**kwargs)
        self.strides = strides
        self.conv = tf.keras.layers.SeparableConv1D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding="same", depthwise_initializer=kernel_initializer,
            pointwise_initializer=kernel_initializer,
            depthwise_regularizer=kernel_regularizer, pointwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, name=f"{self.name}_conv"
        )
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.activation = get_activation(activation)

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv(inputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs)
        return outputs


class ConvModuleLR(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 filters: int = 256,
                 activation: str = "silu",
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(ConvModuleLR, self).__init__(**kwargs)
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")
        self.activation = get_activation(activation)
        self.strides = strides
        self.conv = tf.keras.layers.Conv1D(
            filters=filters, kernel_size=1, strides=strides, padding="same", kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer, name=f"{self.name}_conv"
        )
        self.dsc = tf.keras.layers.DepthwiseConv2D(kernel_size=(kernel_size, 1), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer,
                                                   depth_multiplier=1, dilation_rate=(1, 1), bias_regularizer=bias_regularizer,
                                                   name=f"{self.name}_convDSC")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv(inputs, training=training)
        outputs = tf.expand_dims(outputs, axis=-2)
        outputs = self.dsc(outputs, training=training)
        outputs = tf.squeeze(outputs, axis=-2)
        outputs = self.bn(outputs, training=training)
        outputs = self.activation(outputs)
        return outputs


class SEModule(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size: int = 3,
                 strides: int = 1,
                 filters: int = 256,
                 activation: str = "silu",
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 lrcn=False,
                 **kwargs):
        super(SEModule, self).__init__(**kwargs)

        self.conv = get_conv_module(lrcn, kernel_size=kernel_size, strides=strides,
                                    filters=filters, activation=activation, kernel_initializer=kernel_initializer,
                                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                    name=f"{self.name}_conv_module"
                                    )
        self.activation = get_activation(activation)
        self.fc1 = tf.keras.layers.Dense(filters // 8, name=f"{self.name}_fc1")
        self.fc2 = tf.keras.layers.Dense(filters, name=f"{self.name}_fc2")

    def call(self, inputs, training=False, **kwargs):
        features, input_length = inputs
        outputs = self.conv(features, training=training)

        se = tf.divide(tf.reduce_sum(outputs, axis=1), tf.expand_dims(tf.cast(input_length, dtype=outputs.dtype), axis=1))
        se = self.fc1(se, training=training)
        se = self.activation(se)
        se = self.fc2(se, training=training)
        se = self.activation(se)
        se = tf.nn.sigmoid(se)
        se = tf.expand_dims(se, axis=1)

        outputs = tf.multiply(outputs, se)
        return outputs


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 nlayers: int = 3,
                 kernel_size: int = 3,
                 filters: int = 256,
                 strides: int = 1,
                 residual: bool = True,
                 activation: str = 'silu',
                 alpha: float = 1.0,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 lrcn=False,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)

        self.dmodel = filters
        self.time_reduction_factor = strides
        filters = int(filters * alpha)

        self.convs = []
        for i in range(nlayers - 1):
            self.convs.append(get_conv_module(lrcn, kernel_size=kernel_size, strides=1,
                                              filters=filters, activation=activation, kernel_initializer=kernel_initializer,
                                              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                              name=f"{self.name}_conv_module_{i}"))

        self.last_conv = get_conv_module(lrcn, kernel_size=kernel_size, strides=strides,
                                         filters=filters, activation=activation, kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                         name=f"{self.name}_conv_module_{nlayers - 1}"
                                         )

        self.se = SEModule(lrcn=lrcn, kernel_size=kernel_size, strides=1, filters=filters, activation=activation,kernel_initializer=kernel_initializer,
                           kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                           name=f"{self.name}_se"
                           )

        self.residual = None
        if residual:
            self.residual = get_conv_module(lrcn, kernel_size=kernel_size, strides=strides,
                                            filters=filters, activation="linear",
                                            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,kernel_initializer=kernel_initializer,
                                            name=f"{self.name}_residual"
                                            )

        self.activation = get_activation(activation)

    def call(self, inputs, training=False, **kwargs):
        features, input_length = inputs
        outputs = features
        for conv in self.convs:
            outputs = conv(outputs, training=training)
        outputs = self.last_conv(outputs, training=training)
        input_length = math_util.get_reduced_length(input_length, self.last_conv.strides)
        outputs = self.se([outputs, input_length], training=training)
        if self.residual is not None:
            res = self.residual(features, training=training)
            outputs = tf.add(outputs, res)
        outputs = self.activation(outputs)
        return outputs, input_length


class CnnFeaturizer(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_initializer="glorot_uniform",
                 wave_kernel_size=400,
                 wave_strides=160,
                 wave_filters=80,
                 **kwargs):
        super(CnnFeaturizer, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters=wave_filters, kernel_size=wave_kernel_size,kernel_initializer=kernel_initializer,
                                           strides=wave_strides, padding="same", name=f"{self.name}_conv")

    def call(self, inputs, training=False, **kwargs):
        try:
            outputs = self.conv(inputs, training=training)
            return outputs
        except Exception as e:
            print(" error is :", e)
            print("There is some problem with the choice of cnn featurizer kernel, strides and filters")
            print("Please kindly modify these in config file under the encoder key")
            return None


def get_featurizer(wave_model, kernel_initializer="glorot_uniform",
                   wave_kernel_size=400,
                   wave_strides=160,
                   wave_filters=80,
                   name=None):
    if wave_model:
        return CnnFeaturizer(wave_kernel_size=wave_kernel_size, wave_strides=wave_strides,kernel_initializer=kernel_initializer,
                             wave_filters=wave_filters,
                             name=f"{name}_cnn_featurizer")
    return Reshape(name=f"{name}_reshape")


def get_kernel_initializer(kernel_initializer):
    if kernel_initializer == "orthogonal":
        return tf.keras.initializers.Orthogonal(gain=1.0, seed=None)
    return kernel_initializer


class ContextNetEncoder(tf.keras.Model):
    def __init__(self,
                 blocks: List[dict] = [],
                 alpha: float = 1.0,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_initializer="glorot_uniform",
                 lrcn: bool = False,
                 wave_model: bool = False,
                 wave_kernel_size=400,
                 wave_strides=160,
                 wave_filters=80,
                 **kwargs):
        super(ContextNetEncoder, self).__init__(**kwargs)

        self.kernel_initializer = get_kernel_initializer(kernel_initializer)

        self.wave_model = wave_model
        self.featurizer = get_featurizer(wave_model, wave_kernel_size=wave_kernel_size,
                                         wave_strides=wave_strides, wave_filters=wave_filters,
                                         name=self.name)
        self.feature_output = []
        self.blocks = []
        for i, config in enumerate(blocks):
            self.blocks.append(
                ConvBlock(
                    **config, alpha=alpha, kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, lrcn=lrcn,
                    name=f"{self.name}_block_{i}"
                )
            )

    def call(self, inputs, training=False, **kwargs):
        outputs, input_length, signal = inputs
        if self.wave_model:
            outputs = self.featurizer(signal, training=training)
        else:
            outputs = self.featurizer(outputs)
        for block in self.blocks:
            outputs, input_length = block([outputs, input_length], training=training)

        return outputs

    def call_feature_output(self, inputs, training=False, **kwargs):
        outputs, input_length, signal = inputs
        if self.wave_model:
            outputs = self.featurizer(signal, training=training)
        else:
            outputs = self.featurizer(outputs)
        self.feature_output.append(outputs)
        for i, block in enumerate(self.blocks):
            # if i == len(self.blocks) - 1:
            #     break
            outputs, input_length = block([outputs, input_length], training=training)
            self.feature_output.append(outputs)

        return outputs
