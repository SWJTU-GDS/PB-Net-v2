# Demo Code for Paper:
# [Title]  - "Deep Physiological-Behavioral Representation Learning for Video-Based Hand Gesture Authentication"
# [Author] - Wenwei Song, Xiaorong Gao, Yufeng Zhang, Jinlong Li, Wenxiong Kang, and Zhixue Wang
# [Github] - https://github.com/SWJTU-GDS/PB-Net-v2

import torch
import torch.nn.functional as F
from model.PBNet import Model_PBNet_v2
from loss.loss import AMSoftmax

def feedforward_demo(model, out_dim, is_train=False):

    if is_train:
        # AMSoftmax loss function
        # there are 143 identities in the training set
        criterian_p = AMSoftmax(in_feats=out_dim, n_classes=143)
        criterian_b = AMSoftmax(in_feats=out_dim, n_classes=143)
        criterian_pb = AMSoftmax(in_feats=out_dim * 2, n_classes=143)

    data = torch.randn(2, 64, 3, 224, 224)  #batch, frame, channel, h, w
    data = data.view(-1, 3, 224, 224)  #regard the frame as batch (TSN paradigm)
    id_feature, x_b_norm, x_p_norm, weight = model(data) # feedforward

    if is_train is False:
        # Use the id_feature to calculate the EER when testing
        return id_feature
    else:
        # Use the id_feature, x_b_norm, x_p_norm, weight to calculate loss when training
        label = torch.randint(0, 143, size=(2,))
        loss_p, _ = criterian_p(x_p_norm, label)
        loss_b, _ = criterian_b(x_b_norm, label)
        loss_pb, _ = criterian_pb(id_feature, label)
        loss_rate = F.relu(weight[:, 1] - weight[:, 0]).mean()
        return loss_p + loss_b + loss_pb + loss_rate

if __name__ == '__main__':
    # there are 64 frames in each dynamic hand gesture video
    frame_length = 64
    # the feature dim of last feature map (layer4) from ResNet18 is 512
    feature_dim = 512
    # the identity feature dim
    out_dim = 512

    model = Model_PBNet_v2(frame_length=frame_length, feature_dim=feature_dim, out_dim=out_dim)
    # feedforward_test
    id_feature = feedforward_demo(model, out_dim, is_train=False)
    # feedforward_train
    loss = feedforward_demo(model, out_dim, is_train=True)
    print("Demo is finished!")

