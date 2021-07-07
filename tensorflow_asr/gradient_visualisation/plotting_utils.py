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
import os
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf


def obtain_cmap(color_map):
    return plt.get_cmap(color_map)


def make_directories(current_working_directory_abs, directory_name):
    _directory_abs = os.path.join(current_working_directory_abs, directory_name)
    try:
        os.mkdir(_directory_abs)
    except Exception as e:
        print("--------------", directory_name, "directory already exists-----------------")
        print("--------------The contents will be over-ridden-------------------")
        return _directory_abs

    return _directory_abs


def normalize_gradients(gradient_directory):
    norm_max_g = -1
    normm_max_r = -1

    for index, file in enumerate(sorted(os.listdir(gradient_directory))):
        filename = os.path.join(gradient_directory, file)
        print("filenmes ", filename)
        with open(filename, "rb") as f:
            x_temp = pickle.load(f)

        gradients_check = x_temp["integrated_gradients"]
        random_gradients = x_temp["random_integrated_gradients"]

        for i, j in zip(gradients_check, random_gradients):
            norm_max_g = tf.maximum(tf.norm(i), norm_max_g)
            normm_max_r = tf.maximum(tf.norm(i), normm_max_r)

            print("normm=======", norm_max_g, normm_max_r)

    return norm_max_g, normm_max_r

