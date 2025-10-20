import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import TwoInputSequential, Sub_Adder
from . import definitions as Def

__author__ = "Zhijian Yang"
__copyright__ = "Copyright 2019-2020 The CBICA & SBIA Lab"
__credits__ = ["Zhijian Yang"]
__license__ = "See LICENSE file"
__version__ = "0.1.0"
__maintainer__ = "Zhijian Yang"
__email__ = "zhijianyang@outlook.com"
__status__ = "Development"

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.1)

def define_Linear_Mapping(z_dim, input_dim, hp):
    netG = LMappingGenerator(z_dim, input_dim, hp)
    netG.apply(weights_init)
    return netG

def define_Linear_Discriminator(input_dim, hp):
    netD=LDiscriminator(input_dim, hp)
    netD.apply(weights_init)
    return netD

def define_Linear_Reconstruction(z_dim, input_dim, hp):
    netC = LEncoder(z_dim, input_dim, hp)
    netC.apply(weights_init)
    return netC

def define_Linear_Decomposer(z_dim, input_dim, hp):
    netD = LDecompose(z_dim, input_dim, hp)
    netD.apply(weights_init)
    return netD

def define_Latent_Corr(z_dim):
    netL = LLatentCorr(z_dim)
    netL.apply(weights_init)
    return netL

##############################################################################
# Network Classes
##############################################################################

class LMappingGenerator(nn.Module):
    def __init__(self, z_dim, input_dim, hp, product_layer=Sub_Adder):
        super(LMappingGenerator, self).__init__()

        self.model = TwoInputSequential()
        latent_zx_dim = 0

        #Build the encoding layer:
        for i, layer in enumerate(hp[Def.ENCODING_LAYER]):
            latent_zx_dim = layer[Def.OUT_CHANNELS]

            self.model.append(nn.Linear(
                    layer[Def.IN_CHANNELS] if i > 0 else input_dim,
                    layer[Def.OUT_CHANNELS],
                    bias=layer[Def.BIAS]
                )
            )
            self.model.append(nn.LeakyReLU(
                    layer[Def.RELU_ALPHA],
                    inplace=True
                )
            )

        #Build the Z sampling and sub adder layer (combines X and sampled Z).
        self.model.append(product_layer(
                latent_zx_dim,
                z_dim
            )
        )

        #Build the decoding layer.
        last_index = len(hp[Def.DECODING_LAYER]) - 1
        for i,layer in enumerate(hp[Def.DECODING_LAYER]):
            self.model.append(nn.Linear(
                    layer[Def.IN_CHANNELS],
                    layer[Def.OUT_CHANNELS] if i < last_index else input_dim,
                    bias=layer[Def.BIAS]
                )
            )
            self.model.append(nn.LeakyReLU(
                    layer[Def.RELU_ALPHA],
                    inplace=True
                )
            )

    def forward(self, input_x,input_z):
        return self.model(input_x,input_z)

class LEncoder(nn.Module):
    def __init__(self, z_dim, input_dim, hp):
        super(LEncoder, self).__init__()

        self.model = nn.Sequential()
        for i, layer in enumerate(hp[:-1]):
            self.model.append(nn.Linear(
                    layer[Def.IN_CHANNELS] if i > 0 else input_dim,
                    layer[Def.OUT_CHANNELS],
                    bias=layer[Def.BIAS]
                )
            )
            self.model.append(nn.LeakyReLU(
                    layer[Def.RELU_ALPHA],
                    inplace=True
                )
            )

        #Append the final layer without a ReLU.
        #This should always map to the number of z dimensions.
        self.model.append(nn.Linear(
                hp[-1][Def.IN_CHANNELS],
                z_dim,
                bias=hp[-1][Def.BIAS]
            )
        )

    def forward(self, input_y):
        zt = self.model(input_y)
        return zt

class LDecompose(nn.Module):
    def __init__(self, z_dim, input_dim, hp):
        super(LDecompose, self).__init__()
        
        self.z_dim = z_dim
        self.input_dim = input_dim

        self.model = nn.Sequential()
        last_index = len(hp) - 1
        for i, layer in enumerate(hp):
            self.model.append(nn.Linear(
                    layer[Def.IN_CHANNELS] if i > 0 else input_dim,
                    layer[Def.OUT_CHANNELS] if i < last_index else (z_dim * input_dim),
                    bias=layer[Def.BIAS]
                )
            )
            self.model.append(nn.LeakyReLU(
                    layer[Def.RELU_ALPHA],
                    inplace=True
                )
            )

    def forward(self, input_y):
        z_decompose = self.model(input_y)
        return [z_decompose[:,self.input_dim*i:self.input_dim*(i+1)] for i in range(self.z_dim)]


class LDiscriminator(nn.Module):
    def __init__(self, input_dim, hp):
        super(LDiscriminator, self).__init__()

        self.model = nn.Sequential()
        for i, layer in enumerate(hp[:-1]):
            self.model.append(nn.Linear(
                    layer[Def.IN_CHANNELS] if i > 0 else input_dim,
                    layer[Def.OUT_CHANNELS],
                    bias=layer[Def.BIAS]
                )
            )
            self.model.append(nn.LeakyReLU(
                    layer[Def.RELU_ALPHA],
                    inplace=True
                )
            )

        #Append the final layer without a ReLU.
        #This should always map to 2 (real/fake) classifier.
        self.model.append(nn.Linear(
                hp[-1][Def.IN_CHANNELS],
                2,
                bias=hp[-1][Def.BIAS]
            )
        )

    def forward(self, input_y):
        pred = self.model(input_y)
        return pred

class LLatentCorr(nn.Module):
    def __init__(self, z_dim):
        super(LLatentCorr, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, z_dim, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(z_dim, z_dim, bias=False),
        )

    def forward(self):
        pred =F.tanh(self.model(nn.Parameter(torch.tensor([1.]))))
        return pred
