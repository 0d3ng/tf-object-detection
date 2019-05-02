#  Created by od3ng on 02/05/2019 12:40:17 PM.
#  Project: tf-object-detection
#  File: cv_object_detection.py
#  Email: lepengdados@gmail.com
#  Telegram: @nopriant0

import cv2 as cv
import os

cvNet = cv.dnn.readNetFromTensorflow('models/frozen_inference_graph.pb', 'models/graph.pbtxt')
path_images = "images"
LABELS = open(os.path.join("models", "classes.txt")).read().strip().split("\n")

font_scale = 1
font = cv.FONT_HERSHEY_PLAIN
rectangle_bgr = (255, 255, 255)

for image_name in sorted(os.listdir(path_images)):
    img = cv.imread(os.path.join(path_images, image_name))
    rows = img.shape[0]
    cols = img.shape[1]
    # cvNet.setInput(cv.dnn.blobFromImage(img, 0.017, (300, 300), (127.5, 127.5, 127.5), True, False))
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        class_id = int(detection[1])
        if score > 0.3:
            print("Score: {:.4f}, Class id: {}".format(score, class_id))
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            text = "{} {:.2f}".format(LABELS[class_id], score)

            (text_width, text_height) = cv.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
            text_offset_x = int(left)
            text_offset_y = int(top) - 2
            box_coord = ((text_offset_x, text_offset_y), (text_offset_x + text_width-2, text_offset_y - text_height - 2))

            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 255), thickness=2)
            cv.rectangle(img, box_coord[0], box_coord[1], rectangle_bgr, cv.FILLED)
            cv.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0),
                       thickness=1)
    # cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.imshow(image_name, img)
    cv.waitKey()
