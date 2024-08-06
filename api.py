from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights

from segmentation_model_ai import SegmentationModelAI

origins = ["http://localhost", "http://localhost:8000"]


class Image(BaseModel):
    path: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    pytorch_model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
    model = SegmentationModelAI(pytorch_model)
    yield

    # Can also be used with the converted ONNX model this way
    # onnx_model_path = "./converted_models/deeplabv3_mobilenet_v3_large.onnx"
    # model = SegmentationModelAI(onnx_model_path)


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=origins)


@app.post("/predict")
async def predict(image: Image):
    output = model(image.path)
    return JSONResponse({"output": output.tolist()})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
