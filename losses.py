import torch.nn as nn
import torch


class SMAPELoss(nn.Module):
  def forward(self, input, target):
    return (torch.abs(input - target) / (torch.abs(input) + torch.abs(target) + 1e-2)).mean()

class MAPELoss(nn.Module):
  def forward(self, input, target):
    return (torch.abs(input - target) / (torch.abs(target) + 1e-2)).mean()

