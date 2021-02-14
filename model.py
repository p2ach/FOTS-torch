import numpy as np

import torch
from torch import nn
import math

import pretrainedmodels

from components.shared_convolutions import SharedConvolutions
from components.roi_rotate import ROIRotate
from components.crnn import CRNN

from utils import classes

from bbox import Toolbox


class FOTSModel(nn.Module):
    """
    FOTS model class.

    It contains 3 main components: Shared features extractor,
    text detection branch, text recognition branch.

    - Shared feature extractor uses pretrained ResNet50 (on ImageNet) as backbone.
    - Text detector uses convolutional netwok.
    - Text recognition uses CRNN.
    """

    def __init__(self):
        super().__init__()

        back_bone =  pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        self.shared_conv = SharedConvolutions(back_bone=back_bone)

        n_class = len(classes) + 1  # 1 for "ctc blank" token (0)
        self.recognizer = Recognizer(n_class)
        self.detector = Detector()
        self.roirotate = ROIRotate()

    def to(self, device):
        """Move the FOTS model to given device (GPU/CPU)."""
        self.detector.to(device)
        self.recognizer.to(device)
        self.shared_conv.to(device)
    
    def train(self):
        """Transition the FOTS model to training mode."""
        self.recognizer.train()
        self.detector.train()
        self.shared_conv.train()
    
    def eval(self):
        """Transition the FOTS model to evaluation mode."""
        self.recognizer.eval()
        self.detector.eval()
        self.shared_conv.eval()
    
    @property
    def is_training(self):
        """Check whether the FOTS model is in training mode."""
        return (
            self.detector.training
            and self.recognizer.training
            and self.shared_conv.training
        )
    
    def forward(self, *x):
        """FOTS forward method."""

        images, bboxes, mappings = x

        # Get the device
        if images.is_cuda:
            device = images.get_device()
        else:
            device = torch.device('cpu')

        # Step 1: Extract shared features
        shared_features = self.shared_conv(images)

        # Step 2: Text detection from shared features using detector branch
        per_pixel_preds, loc_features = self.detector(shared_features)

        # Step 3: RoIRotate
        if self.is_training:
            rois, lengths, indices = self.roirotate(shared_features, bboxes[:, :8], mappings)
            # As mentioned in the paper, for training, the ground truth bboxes will be used
            # because the predicted bboxes can harm the training process.
            pred_mapping = mappings
            pred_bboxes = bboxes
        else:
            score = per_pixel_preds.permute(0, 2, 3, 1).detach().cpu().numpy()
            geometry = loc_features.permute(0, 2, 3, 1).detach().cpu().numpy()

            timer = {'net': 0, 'restore': 0, 'nms': 0}

            pred_bboxes = []
            pred_mapping = []
            for i in range(score.shape[0]):
                s = score[i, :, :, 0]
                g = geometry[i, :, :, ]
                bb, _ = Toolbox.detect(score_map=s, geo_map=g, timer=timer)
                bb_size = bb.shape[0]

                if len(bb) > 0:
                    pred_mapping.append(np.array([i] * bb_size))
                    pred_bboxes.append(bb)

            if len(pred_mapping) > 0:
                pred_bboxes = np.concatenate(pred_bboxes)
                pred_mapping = np.concatenate(pred_mapping)
                rois, lengths, indices = self.roirotate(shared_features, pred_bboxes[:, :8], pred_mapping)
            else:
                return per_pixel_preds, loc_features, (None, None), pred_bboxes, pred_mapping, None

        lens = torch.tensor(lengths).to(device)
        preds = self.recognizer(rois, lens)
        preds = preds.permute(1, 0, 2) # B, T, C -> T, B, C

        return per_pixel_preds, loc_features, (preds, lens), pred_bboxes, pred_mapping, indices


class Recognizer(nn.Module):
    """
    Recognition branch of FOTS. This is basically CRNN.
    """

    def __init__(self, n_class):
        super().__init__()
        self.crnn = CRNN(8, 32, n_class, 256)  # h=8 as given in paper.

    def forward(self, rois, lengths):
        return self.crnn(rois, lengths)


class Detector(nn.Module):
    """
    Detector branch of FOTS. This is basically fully convolutions. 
    """

    def __init__(self):
        super().__init__()
        self.conv_score = nn.Conv2d(32, 1, kernel_size = 1)
        self.conv_loc = nn.Conv2d(32, 4, kernel_size = 1)
        self.conv_angle = nn.Conv2d(32, 1, kernel_size = 1)

    def forward(self, shared_features):

        # Dense per pixel prediction scores of words
        # It's probability of each pixel being a positive sample of text
        score = torch.sigmoid(self.conv_score(shared_features))

        # Predict its distances to top, bottom, left, right sides of
        # the bounding box that contains this pixel
        loc = self.conv_loc(shared_features)

        # The loc features are normalized and has values between [0-1]. As these
        # are distances in actual image, scale them to image size.
        loc = torch.sigmoid(loc) * 512

        # Predicts the orientation of bounding box
        angle = self.conv_angle(shared_features)
        # Limit the predicted angle between -45 to 45 degrees
        angle = (torch.sigmoid(angle) - 0.5) * math.pi / 2

        geometry = torch.cat([loc, angle], dim=1)

        return score, geometry
