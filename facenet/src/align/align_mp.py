"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import facenet.src.align.detect_face as fad
import random
import pickle
import cv2
from multiprocessing import Process, Pool
from time import sleep
import time
from PIL import Image

def main_align(lst):

    margin = 20
    image_size=182
    direct = os.path.dirname(os.path.abspath(__file__))
    #sleep(random.random())
    output_dir = direct + "/tmp"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    #facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = lst#facenet.get_dset(lst)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = fad.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    #bounding_boxes_filename = os.path.join(direct, 'bounding_boxes_%05d.txt' % random_key)
    name = {}
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    #random.shuffle(dataset)
    for image_path in dataset:
        nrof_images_total += 1
        print(nrof_images_total)
        
        of = image_path + "_0.png"

        if not os.path.exists(of):
            try:
                bgr_img = cv2.imread(image_path)
                img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
            
                if img.ndim<2:
                    print('Unable to align "%s"' % image_path)
                    #text_file.write('%s\n' % (output_filename))
                    continue
                
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:,:,0:3]

                bounding_boxes, _ = fad.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces>0:
                    dets = bounding_boxes[:,:]
                    img_size = np.asarray(img.shape)[0:2]
                    """
                    if nrof_faces>1:
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        det = det[index,:]
                    """
                    ind = image_path.rfind('/')
                    nam = image_path[ind+1:]
                    for i in range(len(dets)):
                        det = dets[i]
                        if det[4] > 0.95:
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-margin/2, 0)
                            bb[1] = np.maximum(det[1]-margin/2, 0)
                            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                            if  bb[3]-bb[1] > 80 and bb[2] - bb[0] > 80: 
                                #scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                                scaled = Image.fromarray(cropped).resize((image_size,image_size))
                                nrof_successfully_aligned += 1
                                output_filename = output_dir + "/"  + nam +  "_" + str(i) + ".png"
                                scaled.save(output_filename)

                                #text_file.write('%s %d %d %d %d %f\n' % (output_filename, bb[0], bb[1], bb[2], bb[3],det[4]))
                    name[nam] = image_path
                else:
                    print('Unable to align "%s"' % image_path)
                    #text_file.write('%s\n' % (output_filename))
    
    
    print('Total number of images: %d' % nrof_images_total) 
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
            
"""
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    file = open("e:/downloads/mp app/im2txt/im2txt/photos.pk","rb")
    lst = pickle.load(file)
    print(len(lst))
    start = time.time()
    p = Process(target=main_align,args=(lst,))
    p.start()
    p.join()
    
    #find_images("")
    #p = Pool(processes=4)
    #p.map(main_align,lst)
    tm = time.time() - start
    
    #main_align(lst)
"""