import torch
import torch.nn as nn
import timm


### Trainable params: 3,843  initialized from pretrained ###
def ViT_1():
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)
    
    #  Freeze the base parameters
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    #params = model.state_dict()
    #params.keys()
    
    model.norm.weight.requires_grad = True
    model.norm.bias.requires_grad = True
    
    model.head.weight.requires_grad = True
    model.head.bias.requires_grad = True
    
    return model

model=ViT_1()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
model.to(device=DEVICE,dtype=torch.float)
summary(model, (3, 224, 224))

### Trainable params: 85,648,899  initialized Randamly ###
def ViT_2():

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=3)    
    
    return model

model=ViT_2()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
model.to(device=DEVICE,dtype=torch.float)
summary(model, (3, 224, 224))


### Trainable params: 85,648,899  initialized from pretrained ###

def ViT_3():

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=3)    
    
    return model

model=ViT_3()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
model.to(device=DEVICE,dtype=torch.float)
summary(model, (3, 224, 224))
