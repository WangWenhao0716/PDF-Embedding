# PDF-Embedding
[NeurIPS 2024] The official implementation of "Image Copy Detection for Diffusion Models"

![image](https://github.com/WangWenhao0716/PDF-Embedding/blob/main/PDF-Embedding.jpg)


## Dataset

We release the dataset at [HuggingFace](https://huggingface.co/datasets/WenhaoWang/D-Rep). Please follow the instructions here to download the D-Rep dataset in our paper.


## Demonstration

![image](https://github.com/WangWenhao0716/PDF-Embedding/blob/main/match.jpg)


## Installation
```
conda create -n pdfembedding python=3.9
conda activate pdfembedding
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.4.12
```

## Feature Extraction

```python
import requests
import torch
from PIL import Image

from pdf_embedding_extractor import preprocessor, create_model

model_name = 'vit_base'
weight_name = 'vit_exp_563.pth.tar'#'vit_gauss_760.pth.tar'
model = create_model(model_name, weight_name)

url = "https://huggingface.co/datasets/WenhaoWang/AnyPattern/resolve/main/Irises.jpg"
image = Image.open(requests.get(url, stream=True).raw)
x = preprocessor(image).unsqueeze(0)

pdf_features = model.forward_features(x)  # => torch.Size([6, 768])
```

## Matching
```
# Assume we have two pdf_features


```

## Citation
```
@article{wang2024icdiff,
  title={Image Copy Detection for Diffusion Models},
  author={Wang, Wenhao and Sun, Yifan and Tan, Zhentao and Yang, Yi},
  booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=gvlOQC6oP1}
}

```

## Contact

If you have any questions, feel free to contact [Wenhao Wang](https://wangwenhao0716.github.io/) (wangwenhao0716@gmail.com).




