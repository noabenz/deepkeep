from PIL import Image
from torchvision import transforms


class PreprocessStrategy:
    def preprocess(self, image):
        raise NotImplementedError("Preprocess method must be implemented")


class PILPreprocess(PreprocessStrategy):
    def preprocess(self, image):
        print("Using PIL Preprocess")
        preprocess = transforms.Compose([
            transforms.Resize(524),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return preprocess(image).unsqueeze(0)


class NumpyPreprocess(PreprocessStrategy):
    def preprocess(self, image):
        print("Using Numpy Preprocess")
        image = Image.fromarray(image)

        return PILPreprocess().preprocess(image)


class PathPreprocess(PreprocessStrategy):
    def preprocess(self, image):
        print("Using Path Preprocess")
        image = Image.open(image).convert("RGB")

        return PILPreprocess().preprocess(image)
