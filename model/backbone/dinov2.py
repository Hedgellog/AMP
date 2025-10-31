import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self,target_feat, condition):
        super().__init__()
        self.condition = condition
        self.target_feat = target_feat
        self.MLP = cs_token_MLP(3840, 1536)

    def forward(self, features: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:

        B , _ , C= features.shape
        gamma = 1
        beta = 0
        return features * gamma + beta

class cs_token_MLP(nn.Module):
    def __init__(self, input_dim=3840, output_dim=1536):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 3456),      # 3840 -> 6144
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(3456, 2816),           # 6144 -> 8192
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(2816, 2304),          # 8192 -> 10240
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(2304, 1792),          # 8192 -> 10240
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1792, output_dim)     # 10240 -> 12544
        )
    
    def forward(self, x):
        return self.layers(x)

class DINOV2(nn.Module):
    def __init__(self,selected_layers, fileroot, model_size):
        super(DINOV2, self).__init__()
        self.model =  torch.hub.load(fileroot, model_size, source = 'local').cuda()
        self.outfeat_layer = selected_layers
    def base_forward(self, x):
        # frozen parameters
        with torch.no_grad():
            features = self.model.user_forward_features(x, self.outfeat_layer)
            if isinstance(features[0], torch.Tensor):
               features[0] = F.interpolate(
                   features[0], scale_factor = 4, mode = 'bilinear', align_corners = False
               )
               features[1] = F.interpolate(
                   features[1], scale_factor = 2, mode = 'bilinear', align_corners = False
              )
            else:
               features[0][0] = F.interpolate(
                   features[0][0], scale_factor = 4, mode = 'bilinear', align_corners = False
               )
               features[0][1] = F.interpolate(
                   features[0][1], scale_factor = 2, mode = 'bilinear', align_corners = False
               )
        return features
    
def DINOV2s():
    name = 'DINOV2s'
    selected_layers = [3, 5, 7, 11]
    model = DINOV2(selected_layers, '../dinov2', 'dinov2_vits14')
    return name, model

def DINOV2b():
    name = 'DINOV2b'
    selected_layers = [3, 5, 7, 11]
    model = DINOV2(selected_layers,'../dinov2', 'dinov2_vitb14')
    return name, model

def DINOV2l():
    name = 'DINOV2l'
    selected_layers = [7, 11, 15, 23]
    model = DINOV2(selected_layers, '../dinov2', 'dinov2_vitl14')
    return name, model

def DINOV2g():
    name = 'DINOV2g'
    selected_layers = [11, 19, 23, 39]
    model = DINOV2(selected_layers,'../dinov2', 'dinov2_vitg14')
    return name, model

class DINOV3(nn.Module):
    def __init__(self,selected_layers, fileroot, model_size):
        super(DINOV3, self).__init__()
        self.model =  torch.hub.load(fileroot, model_size, source = 'local').cuda()
        # frozen parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.outfeat_layer = selected_layers
        self.cstoken_mlp = cs_token_MLP(3840, 1536)
    def base_forward(self, x):
        token_f, features = self.model.user_forward_features(x, self.outfeat_layer)
        if isinstance(features[0], torch.Tensor):
            features[0] = F.interpolate(
                features[0], scale_factor = 4, mode = 'bilinear', align_corners = False
            )
            features[1] = F.interpolate(
                features[1], scale_factor = 2, mode = 'bilinear', align_corners = False
            )
        else:
            features[0][0] = F.interpolate(
                features[0][0], scale_factor = 4, mode = 'bilinear', align_corners = False
            )
            features[0][1] = F.interpolate(
                features[0][1], scale_factor = 2, mode = 'bilinear', align_corners = False
            )

        return features

def DINOV3_vits16():
    name = 'DINOV3_vits16'
    selected_layers = [3, 5, 7, 11]
    model = DINOV3(selected_layers, '../dinov3', 'dinov3_vits16')
    return name, model

def DINOV3_vits16plus():
    name = 'DINOV3_vits16plus'
    selected_layers = [3, 5, 7, 11]
    model = DINOV3(selected_layers,'../dinov3', 'dinov3_vits16plus')
    return name, model

def DINOV3_vitb16():
    name = 'DINOV3_vitb16'
    selected_layers = [3, 5, 7, 11]
    model = DINOV3(selected_layers, '../dinov3', 'dinov3_vitb16')
    return name, model

def DINOV3_vitl16():
    name = 'DINOV3_vitl16'
    selected_layers = [7, 11, 15, 23]
    model = DINOV3(selected_layers,'../dinov3', 'dinov3_vitl16')
    return name, model

def DINOV3_vitl16plus():
    name = 'DINOV3_vitl16plus'
    selected_layers = [7, 11, 15, 23]
    model = DINOV3(selected_layers,'../dinov3', 'dinov3_vitl16plus')
    return name, model

def DINOV3_vith16plus():
    name = 'DINOV3_vith16plus'
    selected_layers = [8, 15, 23, 31]
    model = DINOV3(selected_layers,'../dinov3', 'dinov3_vith16plus')
    return name, model

def DINOV3_vit7b16():
    name = 'DINOV3_vit7b16'
    selected_layers = [10, 19, 29, 39]
    model = DINOV3(selected_layers,'../dinov3', 'dinov3_vit7b16')
    return name, model