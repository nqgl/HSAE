import torch
import unittest
from .z_sae import AutoEncoder, AutoEncoderConfig

class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        # Create a sample configuration
        cfg = AutoEncoderConfig(
                dict_size = 10,
                seed = 42,
                l1_coeffs = [0.1, 0.2, 0.3],
                lrs = [0.01, 0.02, 0.03],
                device = "cuda")

        self.cfg = AutoEncoderConfig()
        self.model = AutoEncoder(self.cfg)

    def test_forward(self):
        x = torch.randn(10, self.cfg.d_dict)  # Sample input
        output = self.model.forward(x)
        self.assertEqual(output.shape, x.shape)

    def test_get_loss(self):
        loss = self.model.get_loss()
        self.assertIsInstance(loss, torch.Tensor)

    def test_make_decoder_weights_and_grad_unit_norm(self):
        self.model.make_decoder_weights_and_grad_unit_norm()
        self.assertTrue(torch.allclose(self.model.W_dec.norm(dim=-1), torch.ones_like(self.model.W_dec[:, 0, 0])))

    def test_get_version(self):
        version = self.model.get_version()
        self.assertIsInstance(version, int)

    def test_save_and_load(self):
        self.model.save()
        loaded_model = AutoEncoder.load(1)
        self.assertIsInstance(loaded_model, AutoEncoder)

if __name__ == '__main__':
    unittest.main()