import torch
import unittest
from parallel_zsae.z_sae import AutoEncoder, AutoEncoderConfig, post_init_cfg

class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        # Create a sample configuration
        cfg = AutoEncoderConfig(
                dict_size = 10,
                seed = 42,
                l1_coeffs = [0.1, 0.2, 0.3],
                lrs = [0.01, 0.02],
                d_feature= 10,
                device = "cuda",
                batch_size=11)

        self.cfg = post_init_cfg(cfg)

        self.model = AutoEncoder(self.cfg)

    def test_forward(self):
        x = torch.randn(11, self.cfg.d_feature, device="cuda")  # Sample input
        output = self.model.forward(x)
    
        # self.assertEqual(output.shape, x.shape)

    def test_backward_etc(self):
        x = torch.randn(11, self.cfg.d_feature, device="cuda")  # Sample input
        output = self.model.forward(x)
        loss = self.model.get_loss()
        loss.backward()
        self.model.make_decoder_weights_and_grad_unit_norm()
        
    
if __name__ == '__main__':
    unittest.main()