from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random
from collections import OrderedDict

from timm.models import create_model
import models.vit_models
    
__all__ = ['VisionTransformer_', 'vit_tiny', 'vit_small', 'vit_base_query', 'vit_huge', 'vit_large']


class VisionTransformer_(nn.Module):

    def __init__(self, weight, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(VisionTransformer_, self).__init__()
        self.pretrained = True
        self.weight = weight
        self.cut_at_pooling = cut_at_pooling
        self.num_classes = num_classes
        
        if weight == 'tiny':
            vit = create_model(
                'deit_tiny_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif weight == 'small':
            vit = create_model(
                'deit_small_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
        elif weight == 'base':
            '''
            vit = create_model('vit_base_patch16_224_in21k', pretrained=True)
            '''
            vit = create_model(
                'deit_base_query_patch16_224',
                pretrained=True,
                num_classes=1_000,
                drop_rate=0,
                drop_path_rate=0.1,
                drop_block_rate=None,
            )
            
            
        elif weight == 'large':
            vit = create_model('vit_large_patch16_224_in21k', pretrained=True)
            
        elif weight == 'huge':
            vit = create_model('vit_huge_patch14_224_in21k', pretrained=True)
            
        else:
            print("Not implement!!!")
            exit()
        
        #vit.head = nn.Sequential()
        
        self.base = nn.Sequential(
            vit
        ).cuda()
        
        #self.linear = nn.Linear(vit.embed_dim, 512)
        
        #self.classifier = build_metric('cos', vit.embed_dim, self.num_classes, s=64, m=0.35).cuda()
        #self.classifier_1 = build_metric('cos', 512, self.num_classes, s=64, m=0.6).cuda()
        
        #self.projector_feat_bn = nn.Sequential(
        #        nn.Identity()
        #    ).cuda()

        #self.projector_feat_bn_1 = nn.Sequential(
        #        self.linear,
        #        nn.Identity()
        #    ).cuda()
        

    def forward(self, x, y=None):
        x = self.base(x) # [32, 6, 768]
        
        return x#bn_x_512, prob, prob_1

def vit_tiny(**kwargs):
    return VisionTransformer_('tiny', **kwargs)

def vit_small(**kwargs):
    return VisionTransformer_('small', **kwargs)

def vit_base_query(**kwargs):
    return VisionTransformer_('base', **kwargs)

def vit_large(**kwargs):
    return VisionTransformer_('large', **kwargs)

def vit_huge(**kwargs):
    return VisionTransformer_('huge', **kwargs)
