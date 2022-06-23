import enum
import os
import random
from numpy.core.defchararray import encode, join
import torch
import numpy as np
import cv2
import math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.nn.modules.conv import Conv2d
import torchvision.models as models
import torchvision.transforms as T
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from torchvision.models.vgg import vgg19
from torch.autograd import Variable

from . import unet

from enum import Enum
from PIL import Image
"""
import sys
sys.path.append('..')
from modules_v2 import modulated_deform_conv 
"""

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=math.sqrt(5), mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def load_my_state_dict(net,pretrain_net):
    #import pdb;pdb.set_trace()
    own_state = net.state_dict()
    for name, param in pretrain_net.state_dict().items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


def define_D(nclass,channel_size,hidden_size,output_size,dropout_p,max_length, D_ch, nWriter,norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[],load_pretrain_path=None,iam = False):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator    
    The discriminator has been initialized by <init_net>. 
    """
    net = None
    net =CAM_normD(nclass,channel_size,hidden_size,output_size,dropout_p,max_length,D_ch,nWriter,iam)
    init_net(net, init_type, init_gain, gpu_ids)
    
    return net 

class unetDisLoss(nn.Module):
    def __init__(self,target_real_label = 1.0,target_fake_label = 0.0):
        super(unetDisLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
    
    def BCEloss(self,prediciton,targets,prediction_middle,targets_middle):
        loss =  F.binary_cross_entropy_with_logits(prediciton,targets)
        loss_middle =  F.binary_cross_entropy_with_logits(prediction_middle,targets_middle)
        
        return loss,loss_middle

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, prediction_middle,target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        target_tensor_middle = self.get_target_tensor(prediction_middle, target_is_real)
        loss,loss_middle = self.BCEloss(prediction, target_tensor,prediction_middle,target_tensor_middle)
        return loss,loss_middle

class DisLoss(nn.Module):
    def __init__(self,target_real_label = 1.0,target_fake_label = 0.0):
        super(DisLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
    
    def BCEloss(self,prediciton,targets):
        loss =  F.binary_cross_entropy_with_logits(prediciton,targets)
        # loss_middle =  F.binary_cross_entropy_with_logits(prediction_middle,targets_middle)
        
        return loss

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        # target_tensor_middle = self.get_target_tensor(prediction_middle, target_is_real)
        loss = self.BCEloss(prediction, target_tensor)
        return loss

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class CNN(nn.Module):
    def __init__(self, channel_size):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channel_size, 96, 7, stride=1, padding=3),
            nn.BatchNorm2d(96),
            nn.PReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(96, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 160, 3, stride=1, padding=1),
            nn.BatchNorm2d(160),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(160, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.MaxPool2d(2, 2))

    def forward(self, input):
        conv = self.cnn(input)
        return conv

class AttnDecoderRNN_Cell(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1,max_length = 64):
        super(AttnDecoderRNN_Cell, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, encoder_outputs): #input:torch.Size([batch]); hidden:torch.Size([1, 4, 256]);encoder_ouput：torch.Size([64, batch, 256])
        #import pdb;pdb.set_trace()
        bs, c, h, w = encoder_outputs.shape
        T = h*w 
        encoder_outputs = encoder_outputs.reshape(bs, c, T)
        encoder_outputs = encoder_outputs.permute(2,0,1) #torch.Size([64, batch, 256])

        embedded = self.embedding(input) #torch.Size([batch, 256])
        embedded = self.dropout(embedded) #torch.Size([batch, 256])

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1) #torch.Size([batch, 64])
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2)) #torch.Size([batch, 1, 256])

        output = torch.cat((embedded, attn_applied.squeeze(1)), 1) #torch.Size([batch, 512])
        output = self.attn_combine(output).unsqueeze(0) #torch.Size([1,batch, 512])

        output = F.relu(output) #torch.Size([1, batch, 256])
        output, hidden = self.gru(output, hidden)#both:torch.Size([1, batch, 256])

        output = self.out(output[0]) #torch.Size([4, 366])
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self,hidden_size, output_size, dropout_p,max_length,teach_forcing_prob=0.5):
        super(AttnDecoderRNN,self).__init__()
        self.attention_cell =AttnDecoderRNN_Cell(hidden_size, output_size, dropout_p,max_length)
        self.teach_forcing_prob = teach_forcing_prob
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.hidden_size = hidden_size
        self.pil = T.ToPILImage()
        self.tensor = T.ToTensor()
        self.resize = T.Resize(size = (128,128))
   
    def cal(self,image,alpha_map):
        #import pdb;pdb.set_trace()
        alpha_map = alpha_map.cpu().detach().numpy().reshape(8,8)
        alpha_map =((alpha_map /alpha_map.max())*255).astype(np.uint8)
        alpha_map[alpha_map>0]=1
        alpha_map = cv2.resize(alpha_map,(image.shape[3],image.shape[2]))
        alpha_map_tensor = torch.from_numpy(alpha_map).expand_as(image[0]).cuda()
        return alpha_map_tensor
    

    def forward(self,encode,image,text,text_length):
        #import pdb;pdb.set_trace()
        batch_size = image.shape[0]
        decoder_input = text[:,0]
        decoder_hidden = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()        
        
        attention_map_list = []
        loss = 0.0
        teach_forcing = True if random.random() > self.teach_forcing_prob else False
        if teach_forcing:
            for di in range(1, text.shape[1]):
                decoder_output, decoder_hidden, decoder_attention = self.attention_cell(decoder_input, decoder_hidden, encode) #decoder_output:torch.Size([4, 472]); decoder_hidden:torch.Size([1, 4, 256])
                attention_map_list.append(decoder_attention)
                loss += self.criterion(decoder_output, text[:,di])
                decoder_input = text[:,di]
        else:
            for di in range(1, text.shape[1]):
                decoder_output, decoder_hidden, decoder_attention =self.attention_cell(decoder_input, decoder_hidden, encode)
                attention_map_list.append(decoder_attention)
                loss += self.criterion(decoder_output, text[:,di])
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze()
                decoder_input = ni
        
        _,c,h,w=encode.shape
        num_labels = text_length.data.sum()
        new_encode = torch.zeros(num_labels,c,h,w).type_as(encode.data)
        start = 0
        for i,length in enumerate(text_length.data):
            #import pdb;pdb.set_trace()               
            attention_maps = attention_map_list[0:length]
            for j,alpha_map in enumerate(attention_maps):
                #import pdb;pdb.set_trace()
                alpha_map_weight = ((alpha_map[i]-alpha_map[i].min())/(alpha_map[i].max()-alpha_map[i].min())).reshape(1,h,w)
                encode_weight = encode[i]*alpha_map_weight
                new_encode[start] = encode_weight
                start +=1
        
        return loss,new_encode             
        
  

       

class CAM_normD(nn.Module):
    def __init__(self,nclass,channel_size,hidden_size,output_size,dropout_p=0.1,max_length = 64,D_ch =16, nWriter = 1300,iam = False):
        super(CAM_normD,self).__init__()
        self.encoder =CNN(channel_size)
        self.hidden_size = hidden_size
        
        self.decoder_forradical = AttnDecoderRNN(hidden_size, output_size, dropout_p,max_length)
        self.decoderfeat_forradical =nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.InstanceNorm2d(128), nn.PReLU(), nn.AvgPool2d(2, 2),
            nn.Conv2d(128, 64, 3, 1, 1), nn.InstanceNorm2d(128), nn.PReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 16, 3, 1, 1), nn.InstanceNorm2d(16), nn.PReLU(),
            nn.Conv2d(16, 1, 3, 1, 1)
            )        
        
        # self.unetD = unet.Unet_Discriminator(D_ch,input_nc =channel_size)
        self.D = unet.Discriminator(D_ch,input_nc =channel_size)
        
        self.decoder_writerID = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),nn.InstanceNorm2d(256),nn.PReLU(),nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, 1, 1),nn.InstanceNorm2d(512),nn.PReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),nn.InstanceNorm2d(512),nn.PReLU(),nn.MaxPool2d(2,2),
            nn.Conv2d(512, nWriter, 1, 1, 0)
        )
        self.decoder_writerID_forradical = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),nn.InstanceNorm2d(256),nn.PReLU(),nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, 1, 1),nn.InstanceNorm2d(512),nn.PReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),nn.InstanceNorm2d(512),nn.PReLU(),nn.MaxPool2d(2,2),
            nn.Conv2d(512, nWriter, 1, 1, 0)
        )
        
        self.nWriter = nWriter
        

    def initHidden(batch_size,hidden_size):
        result = torch.autograd.Variable(torch.zeros(1, batch_size, hidden_size))
        return result

    def forward(self, image, text_radical, length_radical):
        
        encode = self.encoder(image)
        b, c, _, _ = encode.size() #batch,256
        
        # out, bottleneck_out = self.unetD(image)
        out = self.D(image)
        loss_forradical,new_encode= self.decoder_forradical(encode,image,text_radical,length_radical)
        pred_radical = self.decoderfeat_forradical(new_encode)
        global_writter_ID = self.decoder_writerID(encode).view(b,self.nWriter,-1).mean(2)
        radical_writter_ID = self.decoder_writerID_forradical(new_encode).view(new_encode.size()[0],self.nWriter,-1).mean(2)
            
        # return pred_radical,loss_forradical,out, bottleneck_out,global_writter_ID, radical_writter_ID
        return pred_radical, loss_forradical, out, global_writter_ID, radical_writter_ID
