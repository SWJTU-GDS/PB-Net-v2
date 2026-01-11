# Demo Code for Paper:
# [Title]  - "Deep Physiological-Behavioral Representation Learning for Video-Based Hand Gesture Authentication"
# [Author] - Wenwei Song, Xiaorong Gao, Yufeng Zhang, Jinlong Li, Wenxiong Kang, and Zhixue Wang
# [Github] - https://github.com/SWJTU-GDS/PB-Net-v2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



class Model_PBNet_v2(torch.nn.Module):
    def __init__(self, frame_length, feature_dim, out_dim):
        super(Model_PBNet_v2, self).__init__()
        # there are 64 frames in each dynamic hand gesture video
        self.frame_length = frame_length
        self.out_dim = out_dim  # the feature dim of the two branches

        # load the pretrained ResNet18 for the two branch
        self.P_Branch = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.B_Branch = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # change the last fc with the shape of 512×512
        self.P_Branch.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
        self.B_Branch.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
        # construct the Temporal Conv for the Layer1 and Layer2 in the B_Branch
        self.temp_layer_b1= nn.Conv3d(64, 64, kernel_size=(4, 1, 1), stride=1, padding=(0,0,0), bias=True)
        self.temp_layer_b2 = nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0,0,0), bias=True)
        # construct the Temporal Conv for the Conv1 and Layer1 in the P_Branch
        self.temp_pool_p1 = nn.MaxPool3d(kernel_size=(3,1,1), stride=(3,1,1), padding=0)
        self.temp_layer_p1 = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(3, 1, 1), padding=(0,0,0), bias=True)
        self.temp_layer_p2 = nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(1,0,0), bias=True)
        # initialize the weights and biases of the Temporal Convs to Zero
        nn.init.constant_(self.temp_layer_b1.weight, 0)
        nn.init.constant_(self.temp_layer_b1.bias, 0)
        nn.init.constant_(self.temp_layer_b2.weight, 0)
        nn.init.constant_(self.temp_layer_b2.bias, 0)
        nn.init.constant_(self.temp_layer_p1.weight, 0)
        nn.init.constant_(self.temp_layer_p1.bias, 0)
        nn.init.constant_(self.temp_layer_p2.weight, 0)
        nn.init.constant_(self.temp_layer_p2.bias, 0)
        # reliability proxies (RProxy)
        self.P = nn.Parameter(torch.FloatTensor(out_dim*2, 2))
        nn.init.xavier_normal_(self.P, gain=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # calculate the TSClips
    def getTSClips(self, x):
        # Grouping
        x = x.view((-1, 4)+x.shape[-3:]) # batch*frame/4,4,c,h,w
        # Graying
        x = torch.sum(x, 2) # batch*frame/4,4,h,w
        # temporal sharpening
        tsclips = x[:, :3] * 2 - x[:, 1:] # batch*frame/4,3,h,w
        return tsclips

    # temporal conv for B-Branch (TC)
    def temp_conv_func_b(self, x, conv, temp_pad=(0,0)):
        x = x.reshape(-1, self.frame_length//4, *x.shape[-3:])
        x = x.permute(0, 2, 1, 3, 4)
        x = conv(F.pad(x, (0,0,0,0)+temp_pad, mode='constant', value=0.0)) + x
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(-1, *x.shape[-3:])
        return x

    # temporal conv for P-Branch (TC)
    def temp_conv_func_p(self, x, conv):
        x = x.reshape((-1, (self.frame_length+2)//3, *x.shape[-3:]))
        x = x.permute(0, 2, 1, 3, 4)
        x = conv(x) + x
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape((-1, *x.shape[-3:]))
        return x

    # temporal conv for P-Branch (TMC)
    def temp_adapt_downsample(self, x, pool, conv):
        x = x.reshape((-1, self.frame_length, *x.shape[-3:]))
        x = x.permute(0, 2, 1, 3, 4)
        x = F.pad(x, pad=(0, 0, 0, 0, 0, 2), mode='replicate')
        x_pool = pool(x)
        x_conv = conv(x)
        x = x_pool + x_conv
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape((-1, *x.shape[-3:]))
        return x

    def forward(self, data, label=None):
        # get tailored videos for the two branches
        v_p = data # batch*frame,3,h,w
        v_b = self.getTSClips(data) # batch*frame/4,3,h,w

        # B-Branch v2
        x_b = self.B_Branch.conv1(v_b)
        x_b = self.B_Branch.bn1(x_b)
        x_b = self.B_Branch.relu(x_b) # batch*frame/4,64,h/2,w/2

        x_b = self.B_Branch.layer1[0](x_b) # batch*frame/4,64,h/2,w/2
        x_b = self.temp_conv_func_b(x_b, conv=self.temp_layer_b1, temp_pad=(2,1)) # TC
        x_b = self.B_Branch.layer1[1](x_b) # batch*frame/4,64,h/2,w/2

        x_b = self.B_Branch.layer2[0](x_b) # batch*frame/4,64,h/2,w/2
        x_b = self.temp_conv_func_b(x_b, conv=self.temp_layer_b2, temp_pad=(1,1)) # TC
        x_b = self.B_Branch.layer2[1](x_b) # batch*frame/4,128,h/4,w/4

        # Remainder of ResNet
        for i in range(2, 4):
            layer_name = "layer"+str(i+1)
            layer = getattr(self.B_Branch, layer_name)
            x_b = layer(x_b)

        x_b = self.B_Branch.avgpool(x_b) # batch*frame/4, 512, 1, 1
        x_b = torch.flatten(x_b, 1)
        x_b = self.B_Branch.fc(x_b)
        x_b = x_b.view(-1, self.frame_length//4, self.out_dim) # batch, frame/4, 512
        x_b = torch.mean(x_b, dim=1, keepdim=False) # batch, 512
        x_b_norm = torch.div(x_b, torch.norm(x_b, p=2, dim=1, keepdim=True).clamp(min=1e-12)) # Behavioral features

        # P-Branch v2
        x_p = self.P_Branch.conv1(v_p) # batch*frame,64,h/2,w/2
        # TMC
        x_p = self.temp_adapt_downsample(x_p, pool=self.temp_pool_p1, conv=self.temp_layer_p1) # batch*frame/3,64,h/2,w/2
        x_p = self.P_Branch.bn1(x_p)
        x_p = self.P_Branch.relu(x_p)

        x_p = self.P_Branch.layer1[0](x_p) # batch*frame/3,64,h/2,w/2
        # TC
        x_p = self.temp_conv_func_p(x_p, conv=self.temp_layer_p2) # batch*frame/3,64,h/2,w/2
        x_p = self.P_Branch.layer1[1](x_p)

        # Remainder of ResNet
        for i in range(1, 4):
            layer_name = "layer"+str(i+1)
            layer = getattr(self.P_Branch, layer_name)
            x_p = layer(x_p)

        x_p = self.P_Branch.avgpool(x_p) # batch*frame/3, 512, 1, 1
        x_p = torch.flatten(x_p, 1)
        x_p = self.P_Branch.fc(x_p)
        x_p = x_p.view(-1, (self.frame_length+2)//3, self.out_dim) # batch, frame/3, 512
        x_p = torch.mean(x_p, dim=1, keepdim=False) # batch, 512
        x_p_norm = torch.div(x_p, torch.norm(x_p, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # Physiological features

        # Feature Fusion (APBF v2)
        x_b_norm_d, x_p_norm_d = x_b_norm.detach(), x_p_norm.detach() # block the gradients
        x_weight = torch.cat((x_b_norm_d, x_p_norm_d), dim=-1)
        RProxy = self.P.reshape(2, 512, 2)
        w = torch.div(RProxy, torch.norm(RProxy, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # normalization
        w = w.reshape(1024, 2)
        # cosine similarity with Scale = 30
        weight = torch.mm(x_weight, w) / 2 * 30
        weight_soft = F.softmax(weight, dim=-1)
        # fusion weights
        weight_sqrt = weight_soft.sqrt()
        # reweight according to feature reliability
        x_b_norm_cat = x_b_norm_d * weight_sqrt[:, :1]
        x_p_norm_cat = x_p_norm_d * weight_sqrt[:, 1:]
        # the final identity feature
        id_feature = torch.cat((x_b_norm_cat, x_p_norm_cat), dim=1)

        return id_feature, x_b_norm, x_p_norm, weight


