import torch
import torchvision
import torch.nn as nn
import torch.utils.data
from data_wrapper import TrainCIFAR10
from torch.utils.data import TensorDataset, DataLoader

# checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
RNG = torch.Generator().manual_seed(42)
ngpu = 1
# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
# number of discriminator filters
ndf = 64
nc = 3
batch_size = 128

cifar_data = TrainCIFAR10()

# custom weights_gan initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.mainF = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.mainF, input, range(self.ngpu))
        else:
            output = self.mainF(input)
            return output


def synthetic_forget_data():
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    # load weights_gan to test the model
    netG.load_state_dict(torch.load('../weights_fgan/netG_epoch_24.pth'))
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fs_data = netG(noise)
    label = 9 - get_class(fs_data)
    fs_data = TensorDataset(fs_data, label)
    cifar_forget = cifar_data.forget_set
    fs_data_temp = torch.utils.data.ConcatDataset([fs_data, cifar_forget])
    fs_data = DataLoader(fs_data_temp, batch_size=128)
    return fs_data


def synthetic_retain_data():
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init)
    # load weights_gan to test the model
    netG.load_state_dict(torch.load('../weights_rgan/netG_epoch_24.pth'))
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    rs_data = netG(noise)
    label = get_class(rs_data)
    rs_data = TensorDataset(rs_data, label)
    cifar_retain = cifar_data.retain_set
    rs_data_temp = torch.utils.data.ConcatDataset([rs_data, cifar_retain])
    rs_data = DataLoader(rs_data_temp, batch_size=128)
    return rs_data


def get_class(data):
    net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    net.load_state_dict(torch.load('../weight_classification/classification_weights.pth'))
    outputs = net(data)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


synthetic_retain_data()