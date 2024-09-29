import os
from torch import nn
import torch
import torchvision
import torch.nn.functional as F

def preprocessor(image):
    image = image.convert("RGB")
    image = image.resize((224,224))
    mean, std = [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]

    transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]
    transforms = torchvision.transforms.Compose(transforms)
    
    return transforms(image)

def create_model(name = 'vit_base_query', checkpoint_file = 'vit_exp_563.pth.tar'):
    import models
    model = models.create(name, num_features=0, dropout=0, num_classes=4)
    model = nn.DataParallel(model)
    
    if not os.path.exists(checkpoint_file):
        try:
            os.system("wget https://huggingface.co/datasets/WenhaoWang/D-Rep/resolve/main/%s"%checkpoint_file)
        
        except:
            print("The network is unreachable or the target file does not exists! Please prepare manually!")
    
    ckpt = torch.load(checkpoint_file, map_location='cpu')
    mkg = model.load_state_dict(ckpt,strict = True)
    model = model.eval()
    
    return model.module.base[0]
