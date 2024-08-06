from pathlib import Path

import numpy as np
import torch
from PIL import Image
from onnxruntime import InferenceSession
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

from image_preprocess import PathPreprocess, NumpyPreprocess, PILPreprocess


class BaseModel:
    preprocess_mapping = {str: PathPreprocess(), np.ndarray: NumpyPreprocess(), Image.Image: PILPreprocess()}

    def select_preprocess_strategy(self, image):
        preprocessor = self.preprocess_mapping.get(type(image), None)
        if not preprocessor:
            raise ValueError("Unsupported image type. Must be a file path, numpy array, or PIL Image.")

        return preprocessor

    def preprocess(self, image):
        return self.select_preprocess_strategy(image).preprocess(image)


class TorchModel(BaseModel):
    def __init__(self, model):
        self.model = model

    def __call__(self, image):
        image_tensor = self.preprocess(image)
        self.model.eval()

        with torch.no_grad():
            output = self.model(image_tensor)['out']

        return output


class OnnxModel(BaseModel):
    def __init__(self, model_path):
        self.ort_session = InferenceSession(model_path)

    def __call__(self, image):
        image_tensor = self.preprocess(image)

        ort_inputs = {self.ort_session.get_inputs()[0].name: image_tensor.cpu().numpy()}
        ort_output = self.ort_session.run(None, ort_inputs)

        return ort_output[0]


class SegmentationModelAI:
    def __init__(self, model):
        if isinstance(model, torch.nn.Module):
            self.model = TorchModel(model)
        #  Making an assumption that onnx model path extension is .onnx
        elif isinstance(model, str) and model.endswith('.onnx'):
            if not Path(model).exists():
                raise ValueError(f"Model file path {model} doesn't exist")

            self.model = OnnxModel(model)
        else:
            raise ValueError("Unsupported model type - must be a Pytorch or ONNX model")

    def __call__(self, image):
        try:
            return self.model(image)
        except Exception as e:
            print(f"Model failed segmenting the image because: {e}")


if __name__ == "__main__":
    pytorch_model = deeplabv3_mobilenet_v3_large(
        weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
    onnx_model_path = "./converted_models/deeplabv3_mobilenet_v3_large.onnx"

    model_ai = SegmentationModelAI(pytorch_model)
    pytorch_output = model_ai(r'./images/test_image.png')

    model_ai = SegmentationModelAI(onnx_model_path)
    onnx_output = model_ai(np.array(Image.open(r'./images/test_image.png').convert("RGB")))
