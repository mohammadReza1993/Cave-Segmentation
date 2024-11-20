from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
model = YOLO('best.pt')
image_address = '/data/mreza/CaveSegmentation/data/images/val/00614.jpg'
results = model.predict(image_address, imgsz=[640,960], conf=0.4)
# masks = results.xyxy[0][:, 4].numpy()
# results.show()
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.png')  # save image
