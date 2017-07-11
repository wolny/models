import os
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from PIL import Image
from flask import Flask, request
from flask_restful import Resource, Api
import hashlib
import urllib

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'model_coco'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load a (frozen) Tensorflow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess = tf.Session(graph=detection_graph)

app = Flask(__name__)
api = Api(app)


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return boxes, scores, classes, num_detections


def load_image_into_numpy_array(image_path):
  image = Image.open(image_path)
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def get_top_labels(classes, scores):
    cs = set()
    results = []
    for clazz, score in zip(np.squeeze(classes), np.squeeze(scores)):
        label = category_index[clazz]['name']
        if label not in cs:
            cs.add(label)
            results.append({'label': label, 'score': score.item()})
            # collect top 5 labels
            if len(cs) == 5:
                break
    return results


def img_path(img_url):
    m = hashlib.md5()
    m.update(img_url)
    return '/tmp/' + m.hexdigest() + '.jpg'


class ObjectDetection(Resource):
    def get(self):
        with detection_graph.as_default():
            img_url = request.args.get('imageUrl')
            image_path = img_path(img_url)
            urllib.urlretrieve(img_url, image_path)
            image_np = load_image_into_numpy_array(image_path)
            (boxes, scores, classes, num_detections) = detect_objects(image_np, sess, detection_graph)
            results = get_top_labels(classes, scores)
            print results
            return {'imageUrl': image_path, 'results': results}

api.add_resource(ObjectDetection, '/detect')

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0')
