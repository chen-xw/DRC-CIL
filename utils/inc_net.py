import copy
import torch
import datetime
from torch import nn
from convs.cifar_resnet import resnet32
from convs.resnet import resnet18, resnet34, resnet50
from convs.ucir_cifar_resnet import resnet32 as cosine_resnet32
from convs.ucir_resnet import resnet18 as cosine_resnet18
from convs.ucir_resnet import resnet34 as cosine_resnet34
from convs.ucir_resnet import resnet50 as cosine_resnet50
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import logging
import torch.nn.functional as F


def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'cosine_resnet18':
        return cosine_resnet18(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        return cosine_resnet32()
    elif name == 'cosine_resnet34':
        return cosine_resnet34(pretrained=pretrained)
    elif name == 'cosine_resnet50':
        return cosine_resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module):

    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

class MAFDRC_CIFAR(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False,scale=1):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        nc = [16,32,64]
        nc = [c*scale for c in nc]
        # 1x1 conv can be replaced with more light-weight bn layer
        self.BHO = nn.BatchNorm2d(nc[2])
        self.BHN = nn.BatchNorm2d(nc[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes,cur_task):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
            
            if cur_task==1:
                self.BHO.weight.data = copy.deepcopy(self.BHN.weight.data)
                self.BHO.bias.data = copy.deepcopy(self.BHN.bias.data)
            else:
                self.BHO.weight.data = (copy.deepcopy(self.BHO.weight.data)+copy.deepcopy(self.BHN.weight.data)) /2
                self.BHO.bias.data = (copy.deepcopy(self.BHO.bias.data)+copy.deepcopy(self.BHN.bias.data)) /2

        del self.fc
        self.fc = fc
    
    def update_BH(self,BHO,BHN):
        self.BHO.weight.data = copy.deepcopy(BHO.weight.data)
        self.BHO.bias.data = copy.deepcopy(BHO.bias.data)
        self.BHN.weight.data =copy.deepcopy(BHN.weight.data)
        self.BHN.bias.data = copy.deepcopy(BHN.bias.data)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        new_fs, old_fs = self.BHN(x["half"]), self.BHO(x["half"])
        fs = torch.cat((old_fs,new_fs),dim=0)
        out = self.fc(self.avgpool(fs).view(fs.size(0),-1))
        c = out['logits'].size(0) // 2 
        out.update({"logits":out["logits"],"old_logits":out["logits"][:c,:],"new_logits":out["logits"][c:,:],"fmaps":x["fmaps"]})
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook)
    
    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

class MAFDRC_ImageNet(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False,scale=1):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        self.BHO = nn.BatchNorm2d(512)
        self.BHN = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes,cur_task):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

            if cur_task==1:
                self.BHO.weight.data = copy.deepcopy(self.BHN.weight.data)
                self.BHO.bias.data = copy.deepcopy(self.BHN.bias.data)
            else:
                self.BHO.weight.data = (copy.deepcopy(self.BHO.weight.data)+copy.deepcopy(self.BHN.weight.data)) /2
                self.BHO.bias.data = (copy.deepcopy(self.BHO.bias.data)+copy.deepcopy(self.BHN.bias.data)) /2

        del self.fc
        self.fc = fc
    
    def update_BH(self,BHO,BHN):
        self.BHO.weight.data = copy.deepcopy(BHO.weight.data)
        self.BHO.bias.data = copy.deepcopy(BHO.bias.data)
        self.BHN.weight.data =copy.deepcopy(BHN.weight.data)
        self.BHN.bias.data = copy.deepcopy(BHN.bias.data)

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = (torch.norm(weights[-increment:, :], p=2, dim=1))
        oldnorm = (torch.norm(weights[:-increment, :], p=2, dim=1))
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold/meannew
        print('alignweights,gamma=', gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        new_fs, old_fs = self.BHN(x["half"]), self.BHO(x["half"])
        fs = torch.cat((old_fs,new_fs),dim=0)
        fs = self.avgpool(fs)
        fs = torch.flatten(fs,1)
        out = self.fc(fs)
        c = out['logits'].size(0) // 2 
        out.update({"logits":out["logits"],"old_logits":out["logits"][:c,:],"new_logits":out["logits"][c:,:],"fmaps":x["fmaps"]})
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook)
    
    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias