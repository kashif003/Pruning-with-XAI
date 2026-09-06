# this file will be used to work with the pruning of cnn network


# loading the model
from src.cnn.cnn import Custom_model

model = Custom_model(device="cpu", name = "resnet50")


import torch

tensor  = torch.randn((1,3,224,224))
print("input shape:",tensor.shape)
accuracy , act, grads= model.forward_pass(tensor)
print("output shape:",accuracy.shape, act.keys(), grads.keys())
print("[INFO] Run completed!")

