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

import argparse
import math
import os
import pickle

from tqdm import tqdm

from tensorflow_asr.gradient_visualisation.plotting_utils import make_directories
from tensorflow_asr.utils import env_util

env_util.setup_environment()

DEFAULT_YAML = "/Users/vaibhavsingh/Desktop/FILRCN/contextnet/config.yml"

directory_to_save_gradient_lists = make_directories(os.getcwd(), "gradient_lists")

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRSliceDataset
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer
from tensorflow_asr.models.transducer.contextnet import ContextNet
from tensorflow_asr.optimizers.schedules import TransformerSchedule
from tensorflow_asr.utils import env_util
import tensorflow as tf

tf.keras.backend.clear_session()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})
strategy = env_util.setup_strategy([0])
config = Config(DEFAULT_YAML)

parser = argparse.ArgumentParser(prog=" Compute Gradients Lists")
parser.add_argument("--model_list_folder", "-f", type=str, default=config.learning_config.running_config.checkpoint_directory, help="gives full path to the saved models")
args = parser.parse_args()

model_directory = args.model_list_folder

last_trained_model = os.path.join(model_directory, sorted(os.listdir(model_directory))[-1])

speech_featurizer = TFSpeechFeaturizer(config.speech_config)

text_featurizer = CharFeaturizer(config.decoder_config)
tf.random.set_seed(0)

visualisation_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    **vars(config.learning_config.gradient_dataset_vis_config)
)

batch_size = 1
visualisation_gradient_loader = visualisation_dataset.create(batch_size)
contextnet = ContextNet(**config.model_config, vocabulary_size=text_featurizer.num_classes)
contextnet.make(speech_featurizer.shape)
contextnet.load_weights(last_trained_model, by_name=True)
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
encoder = contextnet.layers[0]

activated_node_list = []
random_activated_node_list = []

for i, j in visualisation_gradient_loader:
    inputs = tf.Variable(i["inputs"])
    inputs_length = tf.Variable(i["inputs_length"])
    signal = tf.Variable(i["signal"])

    encoder_output = encoder.call_feature_output([inputs, inputs_length, signal])
    activated_channels = tf.norm(encoder_output, axis=1)
    activated_node_index = tf.math.argmax(activated_channels, axis=1).numpy()

    activated_node_list.append(activated_node_index[0])
    random_activated_node_list.append(3)


@tf.function
def get_integrated_gradients(encoder, mel_spec, inputs_length, signal, activated_node_index, random_node_index):
    m_steps = 50
    baseline = tf.zeros(shape=mel_spec.shape)
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    print("alphas", alphas.shape)
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    print("alphas_x", alphas_x.shape)
    baseline_x = tf.expand_dims(baseline, axis=0)
    print("baseline ", baseline_x.shape)
    input_x = tf.expand_dims(mel_spec, axis=0)
    print("input", input_x.shape)
    delta = input_x - baseline_x
    interpolated_images = baseline_x + alphas_x * delta
    print("final images", interpolated_images.shape)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(interpolated_images)
        images = tf.expand_dims(interpolated_images, axis=-1)
        print(images.shape)
        encoder_output = encoder.call_feature_output([images, inputs_length, signal])
        gradients = tape.gradient(encoder_output[:, :, activated_node_index], interpolated_images)

        random_gradients = tape.gradient(encoder_output[:, :, random_node_index], interpolated_images)

    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    random_grads = (random_gradients[:-1] + random_gradients[1:]) / tf.constant(2.0)

    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    integrated_random_gradients = tf.math.reduce_mean(random_grads, axis=0)

    return integrated_gradients, integrated_random_gradients


for filename in tqdm(sorted(os.listdir(model_directory))):
    if not filename.endswith(".h5"):
        print(filename)
        continue

    gradient_file = filename.split('.')[0]
    model_name = os.path.join(model_directory, filename)
    print("model being processed now: ", model_name)

    contextnet.load_weights(model_name, by_name=True)
    encoder = contextnet.layers[0]

    m = 0
    images_check = []
    gradients_check = []
    random_gradients_check = []
    for i, j in visualisation_gradient_loader:
        inputs = tf.Variable(i["inputs"])
        inputs_length = tf.Variable(i["inputs_length"])
        signal = tf.Variable(i["signal"])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            encoder_output = encoder.call_feature_output([inputs, inputs_length, signal])
            gradients = tape.gradient(encoder_output[:, :, activated_node_list[m]], inputs)
            random_gradients = tape.gradient(encoder_output[:, :, random_activated_node_list[m]], inputs)

        interated_gradients, random_integrated_gradients = get_integrated_gradients(encoder, tf.squeeze(inputs),
                                                                                    inputs_length, signal,
                                                                                    activated_node_list[m],
                                                                                    random_activated_node_list[m])

        gradients_check.append(interated_gradients)
        random_gradients_check.append(random_integrated_gradients)

        images_check.append(tf.squeeze(inputs))

        print("integrated_gradients shape=========", interated_gradients.shape, random_integrated_gradients.shape)
        m = m + 1

    dd = {'input_image': images_check,
          'integrated_gradients': gradients_check,
          'random_integrated_gradients': random_gradients_check,
          'index_of_activated_node': activated_node_list,
          'index_of_random_node': random_activated_node_list
          }

    file_path_to_save = os.path.join(directory_to_save_gradient_lists, filename)

    with open(file_path_to_save + ".pkl", 'wb') as f:
        pickle.dump(dd, f)
    f.close()

