import numpy as np
import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P



def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            #import pdb;pdb.set_trace()
            num_adain_params += 2*m.num_features
    return num_adain_params

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaIN2d":
            #import pdb; pdb.set_trace()
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]



def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x

def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', act='relu', use_sn=False):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class MLP(nn.Module):
    def __init__(self, nf_in, nf_out, nf_mlp, num_blocks, norm, act, use_sn =False):
        super(MLP,self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp
        self.model.append(LinearBlock(nf_in, nf, norm = norm, act = act, use_sn = use_sn))
        for _ in range((num_blocks - 2)):
            self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))
        self.model.append(LinearBlock(nf, nf_out, norm='none', act ='none', use_sn = use_sn))
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class AdaIN2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(AdaIN2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # if self.affine:
        #     self.weight = nn.Parameter(torch.Tensor(num_features))
        #     self.bias = nn.Parameter(torch.Tensor(num_features))
        # else:
        #     self.weight = None
        #     self.bias = None

        self.weight = None
        self.bias = None


        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "AdaIN params are None"
        N, C, H, W = x.size()
        #import pdb;pdb.set_trace()
        running_mean = self.running_mean.repeat(N)
        running_var = self.running_var.repeat(N)
        x_ = x.contiguous().view(1, N * C, H * W)
        normed = F.batch_norm(x_, running_mean, running_var,
                              self.weight, self.bias,
                              True, self.momentum, self.eps)
        return normed.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(num_features=' + str(self.num_features) + ')'


class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))

  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values;
  # note that these buffers are just for logging and are not used in training.
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv
    return self.weight / svs[0]

class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    def forward_wo_sn(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)



class Attention(nn.Module):
    def __init__(self, ch, which_conv=SNConv2d, name='attention'):
        super(Attention, self).__init__()
        # Channel multiplier
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)
    def forward(self, x, y=None):
        # Apply convs
        #import pdb;pdb.set_trace()
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        # Perform reshapes
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
               preactivation=False, activation=None, downsample=None,):
    super(DBlock, self).__init__()
    #import pdb;pdb.set_trace()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels if wide else self.in_channels
    self.which_conv = which_conv
    self.preactivation = preactivation
    self.activation = activation
    self.downsample = downsample

    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
    self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
    self.learnable_sc = True if (in_channels != out_channels) or downsample else False
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels,
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.preactivation:
      if self.learnable_sc:
        x = self.conv_sc(x)
      if self.downsample:
        x = self.downsample(x)
    else:
      if self.downsample:
        x = self.downsample(x)
      if self.learnable_sc:
        x = self.conv_sc(x)
    return x

  def forward(self, x):
    #import pdb;pdb.set_trace()
    if self.preactivation:
      h = F.relu(x)
    else:
      h = x
    h = self.conv1(h)
    h = self.conv2(self.activation(h))
    if self.downsample:
      h = self.downsample(h)

    return h + self.shortcut(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d,which_bn= nn.BatchNorm2d, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv,self.which_bn =which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    
    def forward(self, x):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GBlock2(nn.Module):
  def __init__(self, in_channels, out_channels,
               which_conv=nn.Conv2d, activation=None,
               upsample=None, skip_connection = True):
    super(GBlock2, self).__init__()

    self.in_channels, self.out_channels = in_channels, out_channels
    self.which_conv = which_conv
    self.activation = activation
    self.upsample = upsample

    # Conv layers
    self.conv1 = self.which_conv(self.in_channels, self.out_channels)
    self.conv2 = self.which_conv(self.out_channels, self.out_channels)
    self.learnable_sc = in_channels != out_channels or upsample
    if self.learnable_sc:
      self.conv_sc = self.which_conv(in_channels, out_channels,
                                     kernel_size=1, padding=0)

    # upsample layers
    self.upsample = upsample
    self.skip_connection = skip_connection

  def forward(self, x):
    h = self.activation(x)
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    h = self.conv1(h)
    #print(h.size())
    h = self.activation(h)
    h = self.conv2(h)
    # may be changed to h = self.conv2.forward_wo_sn(h)
    if self.learnable_sc:
      x = self.conv_sc(x)


    if self.skip_connection:
        out = h + x
    else:
        out = h
    return out

def content_encoder_arch(ch =16,out_channel_multiplier = 1, input_nc = 3):
    arch = {}
    n=2
    arch[128] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2,4,8]],
                                'out_channels' : [item * ch for item in [1,2,4,8,16]],
                                'resolution': [64,32,16,8,4]}
    
    arch[256] = {'in_channels':[input_nc]+[ch*item for item in [1,2,4,8,8]],
                                'out_channels':[item*ch for item in [1,2,4,8,8,16]],
                                'resolution': [128,64,32,16,8,4]}
    return arch


class content_encoder(nn.Module):

    def __init__(self, G_ch=64, G_wide=True, resolution=128,
                             G_kernel_size=3, G_attn='64_32_16_8', n_classes=1000,
                             num_G_SVs=1, num_G_SV_itrs=1, G_activation=nn.ReLU(inplace=False),
                             SN_eps=1e-12, output_dim=1,  G_fp16=False,
                             G_init='N02',  G_param='SN', nf_mlp = 512, nEmbedding = 256, input_nc = 3,output_nc = 3):
        super(content_encoder, self).__init__()

        self.ch = G_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.G_wide = G_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = G_fp16

        if self.resolution == 128:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 256:
            self.save_featrues = [0,1,2,3,4,5]
        
        self.out_channel_nultipiler = 1
        self.arch = content_encoder_arch(self.ch, self.out_channel_nultipiler,input_nc)[resolution]

        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):

            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                             out_channels=self.arch['out_channels'][index],
                                             which_conv=self.which_conv,
                                             wide=self.G_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=nn.AvgPool2d(2))]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        self.init_weights()
        # print('____params____')
        # for name,param in self.named_parameters():
        #     print(name,param.size())
        #import pdb;pdb.set_trace()


    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self,x):
        h = x
        residual_features = []
        residual_features.append(h)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)            
            if index in self.save_featrues[:-1]:
                residual_features.append(h)        
        return h,residual_features


def decoder_textedit_addskip_arch(ch = 64):
    arch = {}
    n = 2
    arch[128] = {'in_channels': [ch * item for item in [32, 8*n, 4*2, 2*2, 1*2, 1]],
                                'out_channels' : [item *ch for item in [8,4,2,1,1]],
                                'out_channels_mix' : [item *ch for item in [1,8,4,2,1,1]],
                                'resolution' : [8,16,32,64,128]}
    arch[256] = {'in_channels': [ch * item for item in [32,16,16,8,4,2,1]],
                                'out_channels': [item * ch for item in [8,8,4,2,1,1]],
                                'resolution' : [8,16,32,64,128,256]}
    
    return arch


class decoder_textedit_addskip(nn.Module):
    def __init__(self, G_ch=64, G_wide=True, resolution=128,
                             G_kernel_size=3, G_attn='64_32_16_8', n_classes=1000,
                             num_G_SVs=1, num_G_SV_itrs=1, G_activation=nn.ReLU(inplace=False),
                             SN_eps=1e-12, output_dim=1,  G_fp16=False,
                             G_init='N02',  G_param='SN', nf_mlp = 512, nEmbedding = 256, input_nc = 3,output_nc = 3):
        super(decoder_textedit_addskip, self).__init__()

        self.ch = G_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.G_wide = G_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = G_fp16
        #nf_mlp 512
        self.nf_mlp = nf_mlp
        #nEmbedding 256
        self.nEmbedding = nEmbedding
        self.adaptive_param_assign = assign_adain_params
        self.adaptive_param_getter = get_num_adain_params

        self.out_channel_multiplier = 1
        self.arch = decoder_textedit_addskip_arch(self.ch)[resolution]

        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)

            self.which_conv_mix = functools.partial(SNConv2d,
                                                    kernel_size=1, padding=0,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
        self.which_bn = functools.partial(AdaIN2d)

        self.blocks = []
        self.mix_blocks = []
        #import pdb;pdb.set_trace()
        for index in range(len(self.arch['out_channels'])):
            upsample_function = functools.partial(F.interpolate, scale_factor=2, mode="nearest") #mode=nearest is default
                                    

            self.blocks += [[GBlock(in_channels = self.arch["in_channels"][index],
                                            out_channels=self.arch['out_channels'][index],
                                            which_conv = self.which_conv,
                                            which_bn=self.which_bn,
                                            activation=self.activation,
                                            upsample=upsample_function)]]
            
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        last_layer = nn.Sequential(nn.InstanceNorm2d(self.arch['out_channels'][-1]),
                                        self.activation,
                                        self.which_conv(self.arch['out_channels'][-1],output_nc))
        
        self.blocks.append(last_layer)
        self.MLP = MLP(self.nEmbedding,self.adaptive_param_getter(self.blocks), self.nf_mlp, 3, 'none', 'relu')
        self.linear_mix = nn.Linear(self.arch["in_channels"][0],self.arch["in_channels"][0])
        
        self.init_weights()
        # print("_____params______")
        # for name, param in self.named_parameters():
        #   print(name, param.size())
    
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)
    
    def forward(self, x, residual_features, style_emd = None, style_fc = None,residual_features_style = None):
        #import pdb;pdb.set_trace()
        adapt_params = self.MLP(style_fc)
        self.adaptive_param_assign(adapt_params,self.blocks)
        h = x
        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index == 0:
                    #h=h
                    #h = torch.cat((h,h),dim=1)

                    h = torch.cat((h,style_emd),dim=1)
                    # h = h.permute(0,2,3,1)
                    # h = self.linear_mix(h)
                    # h = h.permute(0,3,1,2)
                elif index ==1:
                    h = torch.cat((h,residual_features_style[4]),dim=1)
                    #h = torch.cat((h,residual_features[4]),dim=1)
                elif index ==2:                   
                    h = torch.cat((h,residual_features[3]), dim =1)
                elif index ==3:
                    h = torch.cat((h,residual_features[2]), dim =1)
                elif index == 4:
                    h = torch.cat((h,residual_features[1]), dim = 1)
                
            if self.resolution == 256:
                if index == 0:
                    h = torch.cat((h,style_emd),dim=1)
                    h = h.permute(0,2,3,1)
                    h = self.linear_mix(h)
                    h = h.permute(0,3,1,2)
                elif index ==1:
                    h = torch.cat((h,residual_features_style[5]),dim =1)
                elif index ==2:
                    h = torch.cat((h,residual_features[4]),dim =1)
                elif index ==3:
                    h = torch.cat((h,residual_features[3]), dim =1)
                elif index == 4:
                    h = torch.cat((h,residual_features[2]), dim = 1)
                elif index == 5:
                    h = torch.cat((h,residual_features[1]) ,dim =1)
                
            
            for block in blocklist:
                 h = block(h)
                             
        out = self.blocks[-1](h)
        out = torch.tanh(out)

                                        
        return out


def style_encoder_textedit_addskip_arch(ch =16,out_channel_multiplier = 1, input_nc = 3):
    arch = {}
    n=2
    
    arch[128] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2,4,8]],
                                'out_channels' : [item * ch for item in [1,2,4,8,16]],
                                'resolution': [64,32,16,8,4]}
    
    arch[256] = {'in_channels':[input_nc]+[ch*item for item in [1,2,4,8,8]],
                                'out_channels':[item*ch for item in [1,2,4,8,8,16]],
                                'resolution': [128,64,32,16,8,4]}
    return arch


class style_encoder_textedit_addskip(nn.Module):

    def __init__(self, G_ch=64, G_wide=True, resolution=128,
                             G_kernel_size=3, G_attn='64_32_16_8', n_classes=1000,
                             num_G_SVs=1, num_G_SV_itrs=1, G_activation=nn.ReLU(inplace=False),
                             SN_eps=1e-12, output_dim=1,  G_fp16=False,
                             G_init='N02',  G_param='SN', nf_mlp = 512, nEmbedding = 256, input_nc = 3,output_nc = 3):
        super(style_encoder_textedit_addskip, self).__init__()

        self.ch = G_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.G_wide = G_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = G_kernel_size
        # Attention?
        self.attention = G_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = G_activation
        # Initialization style
        self.init = G_init
        # Parameterization style
        self.G_param = G_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = G_fp16

        
        if self.resolution == 128:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 256:
            self.save_featrues = [0,1,2,3,4,5]
        
        self.out_channel_nultipiler = 1
        self.arch = style_encoder_textedit_addskip_arch(self.ch, self.out_channel_nultipiler,input_nc)[resolution]

        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):

            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                             out_channels=self.arch['out_channels'][index],
                                             which_conv=self.which_conv,
                                             wide=self.G_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=nn.AvgPool2d(2))]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        last_layer = nn.Sequential(nn.InstanceNorm2d(self.arch['out_channels'][-1]),
                                        self.activation,
                                        nn.Conv2d(self.arch['out_channels'][-1],self.arch['out_channels'][-1],kernel_size=1,stride=1))
        self.blocks.append(last_layer)
        self.init_weights()
        # print('____params____')
        # for name,param in self.named_parameters():
        #     print(name,param.size())
        #import pdb;pdb.set_trace()


    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    
    def forward(self,x):        
        h = x
        residual_features = []
        residual_features.append(h)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)            
            if index in self.save_featrues[:-1]:
                residual_features.append(h)        
        h = self.blocks[-1](h)
        style_emd = h        
        h = F.adaptive_avg_pool2d(h,(1,1))
        h = h.view(h.size(0),-1)
        
        return style_emd,h,residual_features


def D_unet_arch(ch=64, attention='64',ksize='333333', dilation='111111',out_channel_multiplier=1, input_nc = 3):
    arch = {}

    n = 2

    ocm = out_channel_multiplier

    # covers bigger perceptual fields
    arch[128]= {'in_channels' :       [input_nc] + [ch*item for item in       [1, 2, 4, 8, 16, 8*n, 4*2, 2*2, 1*2,1]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 8,   4,   2,    1,  1]],
                             'downsample' : [True]*5 + [False]*5,
                             'upsample':    [False]*5+ [True] *5,
                             'resolution' : [64, 32, 16, 8, 4, 8, 16, 32, 64, 128],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,11)}}
    return arch

class Unet_Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                             D_kernel_size=3, D_attn='64', n_classes=1000,
                             num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                             SN_eps=1e-12, output_dim=1,  D_fp16=False,
                             D_init='ortho',  D_param='SN',input_nc =3,output_nc = 1):
        super(Unet_Discriminator, self).__init__()


        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16

        if self.resolution==128:
            self.save_features = [0,1,2,3,4]
        elif self.resolution==256:
            self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4
        # Architecture
        self.arch = D_unet_arch(self.ch, self.attention , out_channel_multiplier = self.out_channel_multiplier,input_nc = input_nc  )[resolution]

        #self.unconditional = kwargs["unconditional"]
        self.unconditional=True
        
        if self.D_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            if self.arch["downsample"][index]:
                self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                             out_channels=self.arch['out_channels'][index],
                                             which_conv=self.which_conv,
                                             wide=self.D_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

            elif self.arch["upsample"][index]:
                upsample_function = (functools.partial(F.interpolate, scale_factor=2, mode="nearest") #mode=nearest is default
                                    if self.arch['upsample'][index] else None)

                self.blocks += [[GBlock2(in_channels=self.arch['in_channels'][index],
                                                         out_channels=self.arch['out_channels'][index],
                                                         which_conv=self.which_conv,
                                                         #which_bn=self.which_bn,
                                                         activation=self.activation,
                                                         upsample= upsample_function, skip_connection = True )]]

            
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        last_layer = nn.Conv2d(self.ch*self.out_channel_multiplier,1,kernel_size=1)
        self.blocks.append(last_layer)
       
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        self.linear_middle = self.which_linear(16*self.ch, output_dim)
        
        # print("_____params______")
        # for name, param in self.named_parameters():
        #     print(name, param.size())

        

    
    def forward(self, x, y=None):
        
        h = x
        residual_features = []
        residual_features.append(x)
        # Loop over blocks

        for index, blocklist in enumerate(self.blocks[:-1]):
            if self.resolution == 128:
                if index==6 :
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==7:
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==8:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[1]),dim=1)

            if self.resolution == 256:
                if index==7:
                    h = torch.cat((h,residual_features[5]),dim=1)
                elif index==8:
                    h = torch.cat((h,residual_features[4]),dim=1)
                elif index==9:#
                    h = torch.cat((h,residual_features[3]),dim=1)
                elif index==10:#
                    h = torch.cat((h,residual_features[2]),dim=1)
                elif index==11:
                    h = torch.cat((h,residual_features[1]),dim=1)

            for block in blocklist:
                h = block(h)

            if index in self.save_features[:-1]:
                residual_features.append(h)

            if index==self.save_features[-1]:
                # Apply global sum pooling as in SN-GAN
                h_ = torch.sum(self.activation(h), [2, 3])
                # Get initial class-unconditional output
                bottleneck_out = self.linear_middle(h_)
                # Get projection of final featureset onto class vectors and add to evidence
                if self.unconditional:
                    projection = 0
                else:
                    # this is the bottleneck classifier c
                    emb_mid = self.embed_middle(y)
                    projection = torch.sum(emb_mid * h_, 1, keepdim=True)
                bottleneck_out = bottleneck_out + projection

        out = self.blocks[-1](h)

        if self.unconditional:
            proj = 0
        else:
            emb = self.embed(y)
            emb = emb.view(emb.size(0),emb.size(1),1,1).expand_as(h)
            proj = torch.sum(emb * h, 1, keepdim=True)
            ################
        out = out + proj

        #out = out.view(out.size(0),1,self.resolution,self.resolution)
        out = out.view(out.size(0),-1)
        return out, bottleneck_out

def D_arch(ch=64, attention='64',ksize='333333', dilation='111111',out_channel_multiplier=1, input_nc = 3):
    arch = {}

    n = 2

    ocm = out_channel_multiplier

    # covers bigger perceptual fields
    arch[128]= {'in_channels' :       [input_nc] + [ch * item for item in [1, 2, 4, 8, 16]],
                             'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
                             'downsample' : [True]*5 + [False],
                             'resolution' : [64, 32, 16, 8, 4, 4],
                             'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                                                            for i in range(2,11)}}
    return arch

class Discriminator(nn.Module):

    def __init__(self, D_ch=64, D_wide=True, resolution=128,
                             D_kernel_size=3, D_attn='64', n_classes=1000,
                             num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
                             SN_eps=1e-12, output_dim=1,  D_fp16=False,
                             D_init='ortho',  D_param='SN',input_nc =3,output_nc = 1):
        super(Discriminator, self).__init__()


        # Width multiplier
        self.ch = D_ch
        # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
        self.D_wide = D_wide
        # Resolution
        self.resolution = resolution
        # Kernel size
        self.kernel_size = D_kernel_size
        # Attention?
        self.attention = D_attn
        # Number of classes
        self.n_classes = n_classes
        # Activation
        self.activation = D_activation
        # Initialization style
        self.init = D_init
        # Parameterization style
        self.D_param = D_param
        # Epsilon for Spectral Norm?
        self.SN_eps = SN_eps
        # Fp16?
        self.fp16 = D_fp16



        if self.resolution==128:
            self.save_features = [0,1,2,3,4]
        elif self.resolution==256:
            self.save_features = [0,1,2,3,4,5]

        self.out_channel_multiplier = 1#4
        # Architecture
        self.arch = D_arch(self.ch, self.attention , out_channel_multiplier = self.out_channel_multiplier,input_nc = input_nc  )[resolution]
        self.unconditional=True

        # Which convs, batchnorms, and linear layers to use
        # No option to turn off SN in D right now
        
        if self.D_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                                    eps=self.SN_eps)        
        
        # Prepare model
        # self.blocks is a doubly-nested list of modules, the outer loop intended
        # to be over blocks at a given resolution (resblocks and/or self-attention)
        self.blocks = []

        for index in range(len(self.arch['out_channels'])):

            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                           out_channels=self.arch['out_channels'][index],
                                           which_conv=self.which_conv,
                                           wide=self.D_wide,
                                           activation=self.activation,
                                           preactivation=(index > 0),
                                           downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]

        # Turn self.blocks into a ModuleList so that it's all properly registered.
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # print("_____params______")
        # for name, param in self.named_parameters():
        #     print(name, param.size())  

    
    def forward(self, x, y=None):
        # Stick x into h for cleaner for loops without flow control
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
            # print(h.shape)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(self.activation(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        if y is not None:
            out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out