import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class PruningModule(Module):
    def prune_by_percentile(self, q=5.0, **kwargs):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        # Calculate percentile value
        '''alive_parameters = []
        for name, p in self.named_parameters():
            # We do not prune bias term
            if 'bias' in name or 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive = tensor[:] # flattened array of nonzero values
            alive_parameters.append(alive)

        all_alives = np.concatenate(alive_parameters)
        percentile_value = np.percentile(abs(all_alives), q)
        print(f'Pruning with threshold : {percentile_value}') '''

        # Prune the weights and mask
        # Note that module here is the layer
        # ex) fc1, fc2, fc3
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.percentile(abs(module.weight.data.cpu().numpy()), q)
                module.prune(threshold)
                print(f'Pruning with threshold : {threshold} for layer {name}')

    def prune_by_threshold(self, limit = 0.05):
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                module.prune(limit)
                print(f'Soft pruning with limit : {limit} for layer {name}')

    def prune_by_std(self, s=0.25):
        """
        Note that `s` is a quality parameter / sensitivity value according to the paper.
        According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
        'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'

        I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
        Note : In the paper, the authors used different sensitivity values for different layers.
        """
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                #threshold = np.std(module.weight.data.cpu().numpy()) * s

                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.prune(threshold)


class MaskedLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)

    """
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # Initialize the mask with 1
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def prune(self, threshold, q = 95):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        hard_threshold = np.percentile(abs(tensor), q)
        #new_mask = np.where(abs(tensor) < threshold, 0, mask)
        Mask1 = np.where (tensor>threshold, tensor-threshold, 0)
        Mask2 = np.where (tensor<-threshold, tensor+threshold, 0)
        Mask3 = np.where (abs(tensor) >= hard_threshold, tensor, Mask1 + Mask2)
        tensor = Mask3.astype(np.float32)      
        new_mask =  np.where (Mask3==0, 0, 1).astype(np.float32)
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        tensor = tensor.astype(np.float32)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)




