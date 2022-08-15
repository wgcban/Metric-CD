# -*- coding: utf-8 -*-
"""
Mahalanobis module
--------------------------
"""
import torch
import torch.nn as nn


def MahalanobisDis(x, xhat):
    # This function computes the Mahalanobis Distance between  x and xhat
    # assume shape(x) = shape(xhat) = b, c, h, w
    [b, c, h, w] = x.shape
    
    # Reshaping x and xhat into 1D vectors
    x = x.view(b, c, h*w)
    xhat = xhat.view(b, c, h*w)

    #Computing covariance matrix
    x_mean = torch.mean(x, dim=2, keepdim=True)
    xhat_mean = torch.mean(xhat, dim=2, keepdim=True)
    S = torch.bmm((x-x_mean), (xhat-xhat_mean).permute(0, 2, 1))
    S = torch.linalg.inv(S)
    
    #Computing MahalanobisDis
    DM = torch.mean(torch.bmm(torch.bmm((x-x_mean).permute(0, 2, 1), S), (xhat-xhat_mean)))

    return DM





