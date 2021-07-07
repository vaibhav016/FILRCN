# Copyright 2021 Vaibhav Singh (@vaibhav016)
# Copyright 2021 Dr Vinayak Abrol (_)
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
import argparse
import math
import os
import pickle

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRSliceDataset
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer
from tensorflow_asr.models.transducer.contextnet import ContextNet
from tensorflow_asr.optimizers.schedules import TransformerSchedule
from tensorflow_asr.utils import app_util
from tensorflow_asr.utils import env_util
from tensorflow_asr.gradient_visualisation.plotting_utils import make_directories


def get_weights(net):
    return net.layers[0].get_weights()


def obtain_direction(copy_of_the_weights):
    direction1 = []
    for w in copy_of_the_weights:
        if len(w.shape) == 3:  # check for 3D tensor  or 1d-conv cnn layer  ---- this might be 2D tensor in case of low rank/depthwise seprable tensor
            random_vector = tf.random.normal(w.shape, 0, 1, tf.float32)
            w_norm_tf = tf.norm(tf.reshape(w, (w.shape[0], -1)), axis=1, keepdims=True)[:, :, None]
            d_norm1_tf = tf.norm(tf.reshape(random_vector, (random_vector.shape[0], -1)), axis=1, keepdims=True)[:, :, None]
            random_vector = random_vector * (w_norm_tf / (d_norm1_tf + 1e-10))
            direction1.append(random_vector)
        elif len(w.shape) == 4:
            random_vector = tf.random.normal(w.shape, 0, 1, tf.float32)
            w_norm_tf = tf.norm(tf.reshape(w, (w.shape[0], -1)), axis=1, keepdims=True)[:, :, None, None]
            d_norm1_tf = tf.norm(tf.reshape(random_vector, (random_vector.shape[0], -1)), axis=1, keepdims=True)[:, :, None, None]
            random_vector = random_vector * (w_norm_tf / (d_norm1_tf + 1e-10))
            direction1.append(random_vector)
        else:
            direction1.append(tf.zeros_like(w))
    return direction1


tf.keras.backend.clear_session()
env_util.setup_environment()

DEFAULT_YAML = "/Users/vaibhavsingh/Desktop/FILRCN/contextnet/config.yml"

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})
strategy = env_util.setup_strategy([0])

config = Config(DEFAULT_YAML)
parser = argparse.ArgumentParser(prog=" Compute Loss Lists")
parser.add_argument("--model_list_folder", "-f", type=str, default=config.learning_config.running_config.checkpoint_directory,
                    help="gives full path to the saved models")
args = parser.parse_args()

speech_featurizer = TFSpeechFeaturizer(config.speech_config)
text_featurizer = CharFeaturizer(config.decoder_config)

tf.random.set_seed(0)
test_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    **vars(config.learning_config.test_dataset_config)
)
batch_size = config.learning_config.running_config.batch_size

test_data_loader = test_dataset.create(batch_size)

number_of_points = 9
small_range = -1.0
large_range = 1.0

xcoordinates = np.linspace(small_range, large_range, num=number_of_points)
ycoordinates = np.linspace(small_range, large_range, num=number_of_points)

xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
inds = np.array(range(number_of_points ** 2))
s1 = xcoord_mesh.ravel()[inds]
s2 = ycoord_mesh.ravel()[inds]
coordinate = np.c_[s1, s2]

directory_to_save = make_directories("loss_lists")
model_directory = args.model_list_folder

for filename in tqdm(sorted(os.listdir(model_directory))):
    if not filename.endswith(".h5"):
        print(filename)
        continue
    loss_file = filename.split('.')[0]
    model_name = os.path.join(model_directory, filename)
    print(model_name)
    contextnet = ContextNet(**config.model_config, vocabulary_size=text_featurizer.num_classes)
    contextnet.make(speech_featurizer.shape)
    # contextnet.summary(line_length=100)
    contextnet.load_weights(model_name, by_name=True)
    contextnet.add_featurizers(speech_featurizer, text_featurizer)

    optimizer = tf.keras.optimizers.Adam(
        TransformerSchedule(
            d_model=contextnet.dmodel,
            warmup_steps=config.learning_config.optimizer_config.pop("warmup_steps", 10000),
            max_lr=(0.05 / math.sqrt(contextnet.dmodel))
        ),
        **config.learning_config.optimizer_config
    )

    contextnet.compile(
        optimizer=optimizer,
        steps_per_execution=1,
        global_batch_size=1,
        blank=text_featurizer.blank
    )

    converged_weights = get_weights(contextnet)

    direction1 = obtain_direction(converged_weights)
    direction2 = obtain_direction(converged_weights)

    current_direction1 = direction1
    current_direction2 = direction2
    current_loader = test_data_loader

    loss_list = np.zeros((number_of_points, number_of_points))
    acc_list_greedy_char = np.zeros((number_of_points, number_of_points))
    acc_list_beam_char = np.zeros((number_of_points, number_of_points))
    acc_list_greedy_wer = np.zeros((number_of_points, number_of_points))
    acc_list_beam_wer = np.zeros((number_of_points, number_of_points))
    col_value = 0

    index_list = []
    for count, ind in tqdm(enumerate(inds)):
        index_list.append(count)
        coord = coordinate[count]
        changes = [d0 * coord[0] + d1 * coord[1] for (d0, d1) in zip(current_direction1, current_direction2)]
        k = np.add(changes, converged_weights)
        contextnet.layers[0].set_weights(k)

        loss = contextnet.evaluate(current_loader, batch_size=batch_size, use_multiprocessing=True, workers=8)
        results = contextnet.predict(current_loader, verbose=1, use_multiprocessing=True, workers=8)
        filepath = os.path.join(os.getcwd(), "test.tsv")
        with open(filepath, "w") as openfile:
            openfile.write("PATH\tDURATION\tGROUNDTRUTH\tGREEDY\tBEAMSEARCH\n")
            for i, pred in enumerate(results):
                groundtruth, greedy, beamsearch = [x.decode('utf-8') for x in pred]
                path, duration, _ = test_dataset.entries[i]
                openfile.write(f"{path}\t{duration}\t{groundtruth}\t{greedy}\t{beamsearch}\n")

        res = app_util.evaluate_results(filepath)
        accuracy_gcer = 1 - res['greedy_cer']
        accuracy_gwer = 1 - res['greedy_wer']
        accuracy_bwer = 1 - res['beamsearch_wer']
        accuracy_bcer = 1 - res['beamsearch_cer']

        loss_list[col_value][ind % number_of_points] = loss
        acc_list_greedy_char[col_value][ind % number_of_points] = accuracy_gcer
        acc_list_greedy_wer[col_value][ind % number_of_points] = accuracy_gwer
        acc_list_beam_wer[col_value][ind % number_of_points] = accuracy_bwer
        acc_list_beam_char[col_value][ind % number_of_points] = accuracy_bcer

        ind_compare = ind + 1
        if ind_compare % number_of_points == 0:  col_value = col_value + 1
        # delete the test file which is temporary
        os.remove(filepath)

    data = {'loss_list': [loss_list],
            'greedy_char': [acc_list_greedy_char],
            'greedy_wer': [acc_list_greedy_wer],
            'beam_wer': [acc_list_beam_wer],
            'beam_char': [acc_list_beam_char]
            }

    file_path_to_save_data = os.path.join(directory_to_save, loss_file) + ".pkl"
    with open(file_path_to_save_data, 'wb') as f:
        pickle.dump(data, f)

    f.close()
