import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from onnxruntime import InferenceSession
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

ONNX_MODEL_FILENAME = r'./converted_models/deeplabv3_mobilenet_v3_large.onnx'
IMAGE_PATH = r'./images/test_image.png'


def convert_model_to_onnx(pytorch_model, input_tensor, onnx_filename):
    torch.onnx.export(pytorch_model, input_tensor, onnx_filename)
    print(f"Pytorch model successfully converted to ONNX and saved to {onnx_filename}")


def load_image(filename):
    input_image = Image.open(filename).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(524),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return preprocess(input_image).unsqueeze(0)


def pytorch_model_inference(model, input_image):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        pytorch_output = model(input_image)['out']

    return pytorch_output, time.time() - start_time


def onnx_model_inference(onnx_model_path, input_image):
    start_time = time.time()
    ort_session = InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_image.cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)

    return ort_output[0], time.time() - start_time


def compare_models(pytorch_model, onnx_model_path, image_tensor):
    pytorch_output, pytorch_inference_time = pytorch_model_inference(pytorch_model, image_tensor)
    onnx_output, onnx_inference_time = onnx_model_inference(onnx_model_path, image_tensor)

    l2_diff = np.linalg.norm(pytorch_output.cpu().numpy() - onnx_output)
    print(f"L2 difference between Pytorch and ONNX model outputs: {l2_diff:.3f}")

    print(f"The runtime difference between the two models is "
          f"{abs(pytorch_inference_time - onnx_inference_time):.3f}")

    return pytorch_output, onnx_output


def visualize_results(pytorch_output, onnx_output):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(pytorch_output.squeeze().argmax(0))
    ax[0].set_title("Pytorch Output")

    ax[1].imshow(onnx_output.squeeze().argmax(0))
    ax[1].set_title("ONNX Output")

    plt.show()


if __name__ == '__main__':
    model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
    convert_model_to_onnx(model, torch.randn(1, 3, 512, 512), ONNX_MODEL_FILENAME)

    image_tensor = load_image(IMAGE_PATH)
    original_output, converted_output = compare_models(model, ONNX_MODEL_FILENAME, image_tensor)

    visualize_results(original_output, converted_output)
