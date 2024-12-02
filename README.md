# Cave-Segmentation
This is repository for underwater cave segmentation 

# Training


To train the YOLO model you need to run the train.py file

`python3 train.py`

The `config.yaml` file should be in the same directory as the `train.py` file. Please change the path in the `config.yaml` based on your data address. You need to create two directories, `images` and `labels`. Inside each directory, you need to create to directories, `train` and `val`. Move the corresponding images and labels inside each directory. 

Since `ultralytics` does not accept the `.png` format for the labels, you need to change the format of the labels according to `ultralytics` requirements.

 To change the labels' format, run the following command 
 
 `python3 mask2yolo.py`

 Change the file path based on your data address

 # Test

 To test the model performance, please run
 
 `python3 test.py`

 The `test.py` file and the `best.pt` model should be in the same directory. 
 Also, change the image address in the code.


 # Deployment on Jetson Nano
There are two ways of running the YOLO model on Jetson Nano (My suggestion is the first way)

 You can either use `ultralytics` package or you can manually deploy the model on Jetson Nano.

 To use `ultralytics`, you can run the model inside the docker container provided at https://docs.ultralytics.com/guides/nvidia-jetson/#flash-jetpack-to-nvidia-jetson

It shows all the steps for the model deployment. 


To run the model outside the docker, you need to export it to `.onnx` format first. To do this run

`python3 model_export.py`

Then, you need to copy the model to Jetson Nano and convert it to TensorRT format. Run `onnx2trt.py` on Jetson Nano. 

`Python3 onnx2trt.py path_to_onnx_model path_to_trt_model`

Since you are not using `ultralytics`, you need to take care of outputs. I added a sample (`latency.py`)

`python3 latency.py path_to_trt_model`

This is just a sample code to run the model on Jetson Nano. It does not use actual images. Instead, it uses random inputs. You can change it with actual images. 
There is also the `lightweight_dataframes.py` file that is used to creat a `.csv` file containing the statistics for the model performance. 


 

 
 
