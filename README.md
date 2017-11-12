# Object-detection-sever
This repository contains the tensorflow object detection code for general objects input being an image. 
The frozen graph is a pre-trained model on SSD mobilenet available in the tensorflow repository of object detection API. 
The current implementation is a stand alone server on Flask which can output detection boxes on an image by providing its input path.

1. object_detect_lite.py - Running this script will load the pre-trained frozen graph for object detection as a stand alone server.
    User will only have to provide the url in the browser in the format as mentioned in the url.txt
    
2. frozen_detection_model.pb - This is a frozen graph model trained for general objects and is available in the tensorflow repository 
   for object detection. We load this model to detect various objects in an input image. It is possible to generate this file by training
   our own set of images. The training procedure is very detailed in the object detection API of tensorflow.
   
3. category_index.json - This is a JSON file outlining the common objects that this frozen graph can detect. 
   Each object is assigned a name along with an ID.
   
4. url.txt - Once the object_detect_lite.py is run, a stand alone local server is set up. This text file has a sample link to be used
    which when hit on browser will display the detection boxes for the objects in the image. 
   

   
How to use the file?
   
