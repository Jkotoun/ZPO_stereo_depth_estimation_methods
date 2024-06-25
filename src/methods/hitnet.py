#author: Jiri Vlasak
import numpy as np
import cv2
import glob
import io

from absl import app
from absl import flags
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
import cv2

class HitNet:

    def __init__(self, model_path):
        self.graph_def = tf.GraphDef()
        with open(model_path, 'rb') as f:
            self.graph_def.ParseFromString(f.read())

    def process(self, left, right):
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB).astype(np.float32)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB).astype(np.float32)
        np_images = np.concatenate((left, right), axis=-1) / 255.0
        return self.forward(np_images)


    def forward(self, np_images):
        with tf.Graph().as_default() as default_graph:
            tf.import_graph_def(self.graph_def, name='graph')
            # Setup input-output tensors for the frozen model.
            xl = default_graph.get_tensor_by_name('graph/input:0')
            reference = default_graph.get_tensor_by_name('graph/reference_output_disparity:0')

            # Run the model.
            with tf.Session(graph=default_graph) as sess:
                feed_dict = {xl: np.expand_dims(np_images, 0)}

            
                reference_disparity = sess.run(reference, feed_dict=feed_dict)
            return reference_disparity
        
