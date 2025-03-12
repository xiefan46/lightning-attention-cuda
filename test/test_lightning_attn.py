import unittest
import torch


class TestLightning(unittest.TestCase):


    def test_lightning_attn(self):
        t = torch.tensor([1, 2, 3])
        print(torch.sum(t))