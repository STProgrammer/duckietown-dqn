#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: abdullah
"""
from torch import nn
import torch.nn.functional as F

class QDNModel(nn.Module):
    def __init__(self):
        super(QDNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(64*4*4, 256)
        self.output = nn.Linear(256, 10)
  
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return self.output(x)