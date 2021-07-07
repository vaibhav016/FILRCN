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

import glob
import os

import cv2

def make_directory():
    current_working_directory_abs = os.getcwd()
    video_directory_abs = os.path.join(current_working_directory_abs, "video")
    try:
        os.mkdir(video_directory_abs)
    except Exception as e:
        print("--------------video directory already exists-----------------")
        print("--------------The contents will be over-ridden-------------------")
        return video_directory_abs

    return video_directory_abs

img_array = []
size = (10,10)
figures_working_dir = os.path.join(os.getcwd(), "figs")

fname1 = figures_working_dir+'/log_loss_accuracy/*.png'
fname2 = figures_working_dir+'/log_contour/*.png'

video_directory = make_directory()


for filename1, filename2 in zip(sorted(glob.glob(fname2)), sorted(glob.glob(fname1))):
    print(filename1)
    print(filename2)

    image1 = cv2.imread(filename1)
    image2 = cv2.imread(filename2)
    height, width, layers = image1.shape
    size = (width, height)
    print(size)
    height, width, layers = image2.shape
    size = (width, height)
    print(size)

    vis = cv2.hconcat([image1, image2])
    height, width, layers = vis.shape
    size = (width, height)

    img_array.append(vis)

filename = video_directory + "/contour_video.avi"
print(filename)
out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
