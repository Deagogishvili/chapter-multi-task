import torch
import numpy as np
from PROT.base import DatasetBase


class NSPData(DatasetBase):
    """ dataset .npz file """

    def __init__(self, *args, **kwargs):
        super(NSPData, self).__init__(*args, **kwargs)


class NSPDataOnlyEncoding(DatasetBase):
    """ dataset .npz file with only input encodings"""
    
    def __init__(self, *args, **kwargs):
        super(NSPDataOnlyEncoding, self).__init__(*args, **kwargs)
        self.X = self.X[:, :, :20]
