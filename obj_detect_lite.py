
import numpy as np
import os
from PIL import Image
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse
import sys
import tensorflow as tf
import json
from collections import defaultdict
from _datetime import datetime
import time
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
st_date = datetime.now()

if getattr(sys, 'frozen', False):
    os.chdir(sys._MEIPASS)

app = Flask(__name__)
api = Api(app)

FROZEN_MODEL = 'frozen_detection_model.pb'
CATEGORY_INDEX_FILE = 'category_index.json'
IMAGE_SIZE = (12, 8)
# IMAGE_PATH = os.getcwd() + '/image3.jpg'
# IMAGE_PATH = 'image3.jpg'
NUMBER_OF_DETECTIONS = 20

detection_graph = tf.Graph()
with detection_graph.as_default():
  print("loading model......")
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(FROZEN_MODEL, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def read_json(file_path):
    with open(file_path) as outfile:
        output_data = json.load(outfile)
        return output_data

category_index = read_json(CATEGORY_INDEX_FILE)

with detection_graph.as_default():

  with tf.Session(graph=detection_graph) as sess:
    print("detection begins......")
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # for image_path in TEST_IMAGE_PATHS:

duration = datetime.now() - st_date
print("Duration to load model...", duration)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result
    return timed

# @app.route('/detect')
class ObjectDetect(Resource):

    def show_detected_img(self, img_np, detected_dict):
        for coordinates_lst in detected_dict.values():
            for each_ordinate_set in coordinates_lst:
                image_rgb = Image.fromarray(np.uint8(img_np)).convert('RGB')
                draw = ImageDraw.Draw(image_rgb)
                print("coordinates :: ", each_ordinate_set)
                (left, right, top, bottom) = each_ordinate_set
                # (right, left, bottom, top, ) = each_ordinate_set
                draw.line([(left, top), (left, bottom), (right, bottom),
                           (right, top), (left, top)], width=2, fill='red')
                np.copyto(img_np, np.array(image_rgb))

        # Display the decoded image to verify if it has been decoded
        cv2.imwrite('color_img.jpg', img_np)
        cv2.imshow('Color image', img_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @timeit
    def get(self):
        # print("------------->>>>", IMAGE_PATH)
        parser = reqparse.RequestParser()
        parser.add_argument('image_path', type=str)
        args = parser.parse_args()
        image_path = args['image_path']
        image = Image.open(image_path)
        # print("image :: ", image, type(image), dir(image))
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # print("image_np >> ", image_np, type(image_np), image_np.shape)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # print("image_np_expanded >> ", image_np_expanded,
        #       type(image_np_expanded), image_np_expanded.shape)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        print("boxes ------>>", boxes, type(boxes))
        print("scores ------>>", scores)
        print("classes ------>>", classes)
        print("num ------>>", num)
        # Return the co-ordinates of 3 maximum predictions
        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        image_width, image_height = image_pil.size
        print('\n')
        # print("image_pil :: ", image_pil)
        # print("image_width, image_height :: ", image_width, image_height)

        obj_class_dict = defaultdict()
        for i in range(0, NUMBER_OF_DETECTIONS):
            box = tuple(boxes[i].tolist())
            # left, right, top, bottom - Area defined
            ymin, xmin, ymax, xmax = box
            # obj_ordinates.append(box)
            (xmin_abs, xmax_abs, ymin_abs, ymax_abs) = (xmin * image_width, xmax * image_width,
                                      ymin * image_height, ymax * image_height)
            mapper_class_lst = list(map(int, category_index.keys()))
            if scores[i] > 0.5:
                print("scores[i] :: ", scores[i], i)
                if classes[i] in mapper_class_lst:
                    print("classes[i] :: ", classes[i], i)
                    class_name = category_index[str(classes[i])]['name']
                    print("class_name ------>>", class_name)
            # obj_ordinates.append((xmin_abs, xmax_abs, ymin_abs, ymax_abs))
                    #right, bottom, left, top
                    if class_name not in obj_class_dict:
                        obj_class_dict[class_name] = [(xmin_abs, xmax_abs, ymin_abs, ymax_abs)]
                    else:
                        obj_class_dict[class_name].append((xmin_abs, xmax_abs, ymin_abs, ymax_abs))

        print("image_np_expanded :: ", image_np_expanded)
        self.show_detected_img(image_np, dict(obj_class_dict))

        return jsonify(obj_class_dict)

api.add_resource(ObjectDetect, '/detect')

if __name__ == '__main__':
    # category_index = read_json(CATEGORY_INDEX_FILE)
    # print("-->> ", category_index, type(category_index))
    app.run()








































#C:\Users\uidr8549\AppData\Local\Continuum\Anaconda3\Scripts\pyinstaller --onefile obj_detect_lite.py --add-data frozen_detection_model.pb;. --add-data category_index.json;.


# http://127.0.0.1:5000/detect?image_path=D:\TDSM\MKS_integration_before\TDSM_auth\TF\TF_prac\TF_V1_May12\TF_run\object_detection_TF\obj_detect_make_exe\image3.jpg
#