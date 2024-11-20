from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
model = YOLO('/data/mreza/CaveSegmentation/code/Yolo/runs/segment/train9/weights/best.pt')

results = model.predict('/data/mreza/CaveSegmentation/data/images/val/00614.jpg', imgsz=[640,960], conf=0.4)
# masks = results.xyxy[0][:, 4].numpy()
# results.show()
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.png')  # save image
print(np.unique(results[0].masks[0].cpu().data))