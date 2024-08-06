## DeepKeep Exercise - ML Backend Developer

### Phase one - Torch to ONNX:
The script `convert_pytorch_to_onnx.py` converts the mentioned model to an ONNX model form.
Run this script in order to convert the model, view metrics and visualize the difference between the two
results. 

### Phase two - Wrapper class
The class `SegmentationModelAI` is
initiated with a segmentation model, either a torch model instance, or an ONNX instance.
`SegmentationModelAI` has a `__call__` function which takes an image as input and returns the
model output. The image can be of any reasonable format. Run the main function in `segmentation_model_ai.py`
in order to test the different models and image processors.


### Phase three - FastAPI
To check the api created run `check_api.py` file. This will make a request to the `\predict` endpoint with the path of
the test image as a parameter.