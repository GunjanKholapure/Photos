# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle
import time
import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from multiprocessing import Process,Pool



tf.logging.set_verbosity(tf.logging.INFO)


def main(filenames,upd=False):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               "./model.ckpt-2000000")
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary("word_counts.txt")
  size = len(filenames)
  
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    
    if upd:
      file = open("open.pk","rb")
      result = pickle.load(file)
      file.close()
    else:
      result = {}
    
    cnt = 0
    for filename in filenames:
        try:
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
                captions = generator.beam_search(sess, image)
            
            print(cnt)
            cnt += 1
            prog = "running algo on " + str(size) + " images. %.4f " % (cnt*100/size) + " %"
            #ex.msg(name,  prog )          
                #print("Captions for image %s:" % os.path.basename(filename))
            tmp = {}
            
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                #print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
                if filename not in result:
                  result[filename] = {}
                result[filename]["caption_"+str(i)] = sentence

        except Exception as e:
            print("error")
                
        
    print("done")

    #msg(name,"saving results to open.pk")
    with open("open.pk","wb") as f:
      pickle.dump(result,f)

    

"""
if __name__ == "__main__":
  file = open("photos.pk","rb") 
  photos = pickle.load(file)
  print(len(photos))

  start_time = time.time()

  #main(photos)
  p = Pool(processes=4)
  p.map(main,photos)
  #p = Process(target=main,args=("_",photos,False,))
  #p.start()
  #p.join()
  
  print("program took - %s seconds " % (time.time() - start_time) )


  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
"""
