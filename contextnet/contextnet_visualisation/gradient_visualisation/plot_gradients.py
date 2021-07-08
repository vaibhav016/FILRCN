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
import glob
import os
import pickle

import cv2
import librosa.display
import matplotlib.pyplot as plt

from tensorflow_asr.gradient_visualisation.plotting_utils import obtain_cmap, make_directories, normalize_gradients

parser = argparse.ArgumentParser(prog="Plot Gradients")

parser.add_argument("--cmap", "-c", type=str, default="jet", help="Gives the colormap to the gradient plots")
parser.add_argument("--gradient_list_folder", "-f", type=str, default=None, help="gives full path to the gradient_list")
parser.add_argument("--index_fixed", "-if", type=bool, default=True, help="gives the option to fix the index")

args = parser.parse_args()
current_working_directory = os.getcwd()

# add argparse for cmap, lists, (colorbar range fix)
if not args.gradient_list_folder:
    gradient_directory = os.path.join(current_working_directory, "gradient_lists")
else:
    gradient_directory = args.gradient_list_folder

directory_to_save_plots = make_directories(current_working_directory, "gradient_plots")
directory_to_save_video = make_directories(current_working_directory, "video")

norm_g, norm_r = normalize_gradients(gradient_directory)

def gradient_transformation(gradient, norm):
    # gradient = tf.abs(gradient)
    # gradient = tf.square(gradient)
    # gradient = gradient/norm
    # print(gradient)

    return gradient.numpy().T

def plot_gradients_images(directory_to_save_plots):
    for index, file in enumerate(sorted(os.listdir(gradient_directory))):
        filename = os.path.join(gradient_directory, file)
        print("gradient list file =>  ",filename)
        with open(filename, "rb") as f:
            x_temp = pickle.load(f)

        name = file.split('.')[0].split('_')[-1]

        images_check = x_temp['input_image']
        gradients_check = x_temp["integrated_gradients"]
        random_gradients = x_temp["random_integrated_gradients"]
        activated_node_list = x_temp["index_of_activated_node"]
        random_node_list = x_temp["index_of_random_node"]

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(25, 14), facecolor=(1, 1, 1))
        fig.suptitle(' Epoch ' + str(index), fontsize=16)
        for i, j in enumerate(zip(images_check, gradients_check)):
            # this is to obtain other samples from visualisation data(it has 6 files- even=>male, and odd=>female
            # if i<4:
            #     continue
            print(images_check[i+1].shape)
            print(gradients_check[i].shape)

            ax[0][0].set(title=" Log Mel spectrogram for male voice")
            img = librosa.display.specshow(images_check[i].numpy().T, sr=16000,
                                           fmax=8000, x_axis='time', y_axis='mel', ax=ax[0][0], alpha=1)
            plt.colorbar(img, ax=ax[0][0])
            ax[0][0].label_outer()

            title = " Gradient Attribution for filter " + str(activated_node_list[i])
            ax[0][1].set(title=title)
            img2 = librosa.display.specshow(gradient_transformation(gradients_check[i], norm_g), sr=16000,
                                            fmax=8000, x_axis='time', y_axis='mel',cmap=obtain_cmap(args.cmap), ax=ax[0][1],  alpha=1)
            if args.index_fixed:
                img2.set_clim(vmin=-0.1, vmax=0.1)
            plt.colorbar(img2, ax=ax[0][1])
            ax[0][1].label_outer()

            title = " Gradient Attribution for filter " + str(random_node_list[i])
            ax[0][2].set(title=title)
            img2 = librosa.display.specshow(gradient_transformation(random_gradients[i], norm_r), sr=16000,
                                            fmax=8000, x_axis='time', y_axis='mel', cmap=obtain_cmap(args.cmap),ax=ax[0][2], alpha=1)
            if args.index_fixed:
                img2.set_clim(vmin=-0.1, vmax=0.1)
            plt.colorbar(img2, ax=ax[0][2])
            ax[0][2].label_outer()

    ############################################################  2nd row of female#######################################################

            ax[1][0].set(title=" Log Mel spectrogram for female voice")
            img = librosa.display.specshow(images_check[i + 1].numpy().T, sr=16000,
                                           fmax=8000, x_axis='time', y_axis='mel', ax=ax[1][0], alpha=1)
            plt.colorbar(img, ax=ax[1][0])
            ax[1][0].label_outer()



            img2 = librosa.display.specshow(gradient_transformation(gradients_check[i+1], norm_g), sr=16000,
                                            fmax=8000, x_axis='time', y_axis='mel', cmap=obtain_cmap(args.cmap), ax=ax[1][1],  alpha=1)
            if args.index_fixed:
                img2.set_clim(vmin=-0.1, vmax=0.1)
            plt.colorbar(img2, ax=ax[1][1])
            ax[1][1].label_outer()

            img2 = librosa.display.specshow(gradient_transformation(random_gradients[i+1], norm_r), sr=16000,
                                            fmax=8000, x_axis='time', y_axis='mel', cmap=obtain_cmap(args.cmap), ax=ax[1][2], alpha=1)
            if args.index_fixed:
                img2.set_clim(vmin=-0.1, vmax=0.1)
            plt.colorbar(img2, ax=ax[1][2])
            ax[1][2].label_outer()


            # this break means that only 2 images(male and female) will be displayed in a plot.
            # This is not a sanity break. Its purposeful.
            break

        plt.savefig(directory_to_save_plots + "/Grad" + name +".png")

def make_videos_from_images(directory_to_save_video):
    img_array = []
    size = (10, 10)



    fname1 = os.path.join(os.getcwd(), "gradient2_plots") + "/*.png"
    # fname1 = '/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/contextnet_visualisation/gradient_visualisation/grad_vis_4/*.png'

    for filename1 in (sorted(glob.glob(fname1))):
        print(filename1)
        image1 = cv2.imread(filename1)
        height, width, layers = image1.shape
        size = (width, height)
        print(size)
        img_array.append(image1)

    out = cv2.VideoWriter(directory_to_save_video + "/gradient_vis.avi", cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    plot_gradients_images(directory_to_save_plots)
    make_videos_from_images(directory_to_save_video)


