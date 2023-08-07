import logging
import numpy as np
from tqdm import tqdm
import torch
import math
import copy
import os
import errno
import os.path as osp
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import MAFDRC_CIFAR
from utils.toolkit import tensor2numpy

EPSILON = 1e-8
softmax=nn.Softmax(dim=1)


class Mafdrc(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._snet = None
        self._network = MAFDRC_CIFAR(args['convnet_type'], False,scale=1)

    def after_task(self):
        self._known_classes = self._total_classes
        if self.args["pretrain"] is False:
            self._old_network = self._network.copy().freeze()
            save_checkpoint({
                'state_dict': self._network.state_dict(),
                'seed': self.args['seed'],
                'task': self._cur_task,
            },fpath=osp.join(self.args['save_path'],'checkpoint_'+self.args["dataset"]+'_'+str(self.args['seed'])+'_'+'task'+str(self._cur_task+1)+'.pth.tar'))
            logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.data_manager = data_manager
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes,self._cur_task)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args["num_workers"])

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def incremental_test(self, data_manager):
        self._cur_task +=1
        self.data_manager = data_manager
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes,self._cur_task)

        logging.info('Evaluation on {}-{}'.format(self._known_classes, self._total_classes))

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args["num_workers"])
        checkpoint = torch.load("./pretrain/"+'checkpoint_'+self.args["dataset"]+'_'+str(self.args['seed'])+'_'+'task'+str(self._cur_task+1)+'.pth.tar')
        self._network.load_state_dict(checkpoint["state_dict"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task==0:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
                )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"])    
            self._init_train(train_loader,test_loader,optimizer,scheduler)
        else:
            # Model Adaption
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
                )), lr=self.args["lr"], momentum=0.9, weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["adaptation_epochs"])
            self._model_adaptation(train_loader, test_loader, optimizer, scheduler)

            # Model Fusion
            self._model_fusion(test_loader)

    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["init_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)['new_logits']

                loss=F.cross_entropy(logits,targets) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

            if epoch%5==0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f},'.format(
                self._cur_task, epoch+1, self.args["init_epochs"], losses/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["init_epochs"], losses/len(train_loader), train_acc,  test_acc)
            prog_bar.set_description(info)

        logging.info(info)


    def _model_adaptation(self, train_loader, test_loader, optimizer, scheduler):        
        if len(self._multiple_gpus) > 1:
            for param in self._network.module.BHO.parameters():
                param.requires_grad = False
        else:
            for param in self._network.BHO.parameters():
                param.requires_grad = False
        prog_bar = tqdm(range(self.args["adaptation_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)
                logitN,logitO = logits["new_logits"], logits["old_logits"]

                logit_S = logitN+logitO
                loss_clf=F.cross_entropy(logit_S[:,self._known_classes:],(targets-self._known_classes))
                loss=loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logit_S, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["adaptation_epochs"], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["adaptation_epochs"], losses/len(train_loader), train_acc)
            prog_bar.set_description(info)

        if len(self._multiple_gpus) > 1:
            for param in self._network.module.BHO.parameters():
                param.requires_grad = True
        else:
            for param in self._network.BHO.parameters():
                param.requires_grad = True
        logging.info(info)
        


    def _model_fusion(self, test_loader): 
        # # only for B50 setting
        # if self._cur_task == 0:
        #     self.factor = 0
        # else:
        #     self.factor = math.sqrt(self._total_classes / (self._total_classes - self._known_classes))

        self._snet = MAFDRC_CIFAR(self.args['convnet_type'], False,scale=1)
        self._snet.update_fc(self._total_classes,self._cur_task)
        if len(self._multiple_gpus) > 1:
            self._snet.update_BH(self._network.module.BHO, self._network.module.BHN)
        else:
            self._snet.update_BH(self._network.BHO, self._network.BHN)
        if len(self._multiple_gpus) > 1:
            self._snet = nn.DataParallel(self._snet,self._multiple_gpus)
        self._snet.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._snet.module.convnet.load_state_dict(copy.deepcopy(self._network.module.convnet.state_dict()))
            self._snet.module.copy_fc(copy.deepcopy(self._network.module.fc))
        else:
            self._snet.convnet.load_state_dict(copy.deepcopy(self._network.convnet.state_dict()))
            self._snet.copy_fc(copy.deepcopy(self._network.fc))
        train_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train',appendent=self._get_memory())
        strain_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args["num_workers"])
        optimizer = optim.SGD(filter(
            lambda p: p.requires_grad, self._snet.parameters()), lr=self.args["lr"], momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["fusion_epochs"])

        logit_adjustments = Logit_adjustment(self,strain_loader,self.args['tro'])
        self._network.eval()
        self._old_network.eval()

        for param in self._snet.BHO.parameters():
            param.requires_grad = False
        self._snet.BHO.eval()

        prog_bar = tqdm(range(self.args["fusion_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._snet.train()
            losses=0.
            correct, total=0, 0
            for i, (_, inputs, targets) in enumerate(strain_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                dark_logits = self._snet(inputs)

                # # Only for B50 setting
                # fmaps=dark_logits['fmaps']

                dark_logitN, dark_logitO = dark_logits["new_logits"], dark_logits["old_logits"]
                
                with torch.no_grad():
                    new_logits = self._network(inputs)
                    new_logitN, new_logitO = new_logits["new_logits"], new_logits["old_logits"]
                    old_logits = self._old_network(inputs)
                    old_logitN, old_logitO = old_logits["new_logits"], old_logits["old_logits"]

                    # # Only for B50 setting
                    # old_fmaps=old_logits["fmaps"]

                labelO = F.one_hot(targets,self._total_classes)[:,:self._known_classes].sum(dim=1)
                labelN = F.one_hot(targets,self._total_classes).sum(dim=1)
                loss_ice = (crossEntropy(softmax,dark_logitO,targets,labelO,self._total_classes,self._known_classes) \
                                +crossEntropy(softmax,dark_logitN,targets,labelN,self._total_classes,self._known_classes)) / (labelO.sum()+labelN.sum()).float()
                
                dark_logit_S = dark_logitN+dark_logitO+logit_adjustments
                loss_fce = F.cross_entropy(dark_logit_S,targets) 

                loss_old = _KD_loss(dark_logit_S[:,:self._known_classes],(old_logitN+old_logitO),self.args["T"]) 
                loss_new = _KD_loss(dark_logit_S[:,self._known_classes:],(new_logitN+new_logitO)[:,self._known_classes:],self.args["T"])

                # # Only for B50 Setting
                # spatial_loss=pod_spatial_loss(fmaps, old_fmaps)* self.factor * 5
                loss = self.args["alpha"]*loss_ice +(1-self.args["alpha"])*loss_fce + self.args["beta1"]*loss_old + self.args["beta2"]*loss_new#+spatial_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(dark_logit_S, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._snet, test_loader)
                info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["fusion_epochs"], losses/len(strain_loader), train_acc, test_acc)
            else:
                info = 'SNet: Task {}, Epoch {}/{} => Loss {:.3f},  Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["fusion_epochs"], losses/len(strain_loader),  train_acc)
            prog_bar.set_description(info)


        for param in self._snet.BHO.parameters():
            param.requires_grad = True
        self._snet.BHO.train()

        if self._cur_task > 0:
            self._network = self._snet
        logging.info(info)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return -1*torch.mul(soft, pred).sum()/pred.shape[0]

def crossEntropy(softmax, logit, label, weight,num_classes,known_class):
    target = F.one_hot(label, num_classes)
    loss = - (weight * (target * torch.log(softmax(logit)+1e-7)).sum(dim=1)).sum()
    return loss

def Logit_adjustment(self,train_loader,tro):
    """compute the base probabilities"""

    label_freq = {}
    for i , (_,inputs,target) in enumerate(train_loader):
        target = target.to(self._device)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key,0)+1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(self._device)

    return adjustments

def save_checkpoint(state,fpath=''):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state,fpath)
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def pod_spatial_loss(old_fmaps, fmaps, normalize=True):
    '''
    a, b: list of [bs, c, w, h]
    '''
    loss = torch.tensor(0.).to(fmaps[0].device)
    for i, (a, b) in enumerate(zip(old_fmaps, fmaps)):
        assert a.shape == b.shape, 'Shape error'

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
        b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
        a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
        b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]

        a = torch.cat([a_h, a_w], dim=-1)
        b = torch.cat([b_h, b_w], dim=-1)

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(fmaps)