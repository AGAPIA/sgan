import os
import time
import torch

g_device = None
def initDevice(use_gpu):
    global g_device
    g_device = torch.device("cuda:0" if (torch.cuda.is_available() and use_gpu == 1) else "cpu")

def getDevice():
    return g_device