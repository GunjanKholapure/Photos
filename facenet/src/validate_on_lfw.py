"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
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

import tensorflow as tf
import numpy as np
import argparse
import facenet.src.facenet as face
#import facenet as face
#import lfw
import time
import pickle
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from multiprocessing import Process,Pool


def main_embed(upd=False):
    start_time = time.time()
    lfw_batch_size = 100
    direct = os.path.dirname(os.path.abspath(__file__))
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            #pairs = lfw.read_pairs(os.path.expanduser(args.--lfw_pairsrs))

            # Get the paths for the corresponding images
            #paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
            # Load the model
            pth =  direct + "/align/tmp"
            print(pth)
            paths = os.listdir(pth)
            print(len(paths))
            for i in range(len(paths)):
                paths[i] = direct + "/align/tmp/" + paths[i]
            #print(paths[0])
            
            model_dir = direct + "/20170216-091149"
            print('Model directory: %s' % model_dir)
            meta_file, ckpt_file = face.get_model_filenames(model_dir)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            face.load_model(model_dir, meta_file, ckpt_file)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
            #print(embedding_size)
        
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on images')
            batch_size = lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            
            
            if upd:
                file = open("pte.pk","rb")
                ind_embed = pickle.load(file)
                file.close()
                upd_embed = {}
            else:
                ind_embed = {}
            
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = face.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                for i in range( len(paths_batch) ):
                    ind1 = paths_batch[i].rfind('/')
                    ind2 = paths_batch[i].rfind('_')
                    photo = paths_batch[i][ind1+1:ind2]
                    ind_photo = paths_batch[i][ind1+1:]
                    ind_embed[ind_photo] = emb_array[start_index+i]
                    if upd:
                        upd_embed[ind_photo] = emb_array[start_index+i]
            
            
            with open("pte.pk","wb") as f:
                pickle.dump(ind_embed,f)
            if upd:
                with open("ue.pk","wb") as f:
                    pickle.dump(upd_embed,f)

            print(time.time() - start_time)
            print("done")

"""
            tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array, 
                actual_issame, nrof_folds=args.lfw_nrof_folds)

            print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

            auc = metrics.auc(fpr, tpr)
            print('Area Under Curve (AUC): %1.3f' % auc)
            eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            print('Equal Error Rate (EER): %1.3f' % eer)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
"""