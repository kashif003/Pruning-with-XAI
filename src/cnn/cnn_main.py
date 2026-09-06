# this file will be used to work with the pruning of cnn network


# loading the model

from torchvision.models import resnet50

def load_resnet50_model(name="ResNet50_Weights.DEFAULT"):
    return resnet50(name)

model = load_resnet50_model()

from src.utils import get_img_tensor
from src.get_score.get_baseline_accuracy import validate

accuracy = validate(model, "cpu")
print(accuracy)
print(model)
print("[INFO] Run completed!")

