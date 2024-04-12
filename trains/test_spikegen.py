import unittest
import torch
from spikegen import rate, rate_conv

class TestSpikeGeneration(unittest.TestCase):

    def test_spike_train_generation(self):
        inputs = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float).requires_grad_()
        num_steps = 5
        output = rate(inputs, num_steps=num_steps)

        self.assertEqual(output.size(), (num_steps, len(inputs))) # expected time dimension prepended

        # check if all elements are either 0 or 1 (valid spike values)
        self.assertTrue(torch.all((output == 0) | (output == 1)).item())

    def test_backpropagation(self):
        inputs = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float).requires_grad_()
        outputs = rate_conv(inputs)

        outputs.sum().backward() # create gradients

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.all(inputs.grad != 0))

if __name__ == '__main__':
    unittest.main()
