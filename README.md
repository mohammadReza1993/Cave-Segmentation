# Cave-Segmentation

**Traning**


to train the YOLO model you need to run the train.py file

`python3 train.py`

The `config.yaml` file should be in the same directory as the `train.py` file. Please change the path in the `config.yaml` based on you data address.

Since `ultralytics` does not accept the `.png` format for the labels, you need to change the format of the labels according to `ultralytics` requirements.

 To change the labels' format, run the following command 
 
 `python3 mask2yolo.py`

 Change the file path based on your data address

 **Test**

 To test the model performance, please run
 
 `python3 test.py`

 The `test.py` file and the `best.py` model should be in the same directory. 
 Also, change the image address in the code.
 
