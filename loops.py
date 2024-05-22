from __future__ import print_function, division
from collections import OrderedDict

import sys
import time
import torch
import torch.nn as nn

from util import AverageMeter, accuracy

from distillation import TDALoss

import wandb


def train_vanilla(epoch, train_loader, module_list, criterion, optimizer, opt):
    """vanilla training"""
    for module in module_list:
        module.train()

    model = module_list[0]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

            wandb.log({"train": {"time": batch_time.avg, "loss": losses.avg, "acc": top1.avg}})

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, ripsnet_t=None, criterion_tda=None, rips_pool_t=None, rips_pool_s=None, rips_1dconv=None):
    """One epoch distillation"""

    for module in module_list:
        module.train()

    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for idx, data in enumerate(train_loader):

        input, target, index = data
        data_time.update(time.time() - end)
        
        preact = False
        with torch.no_grad():
            if opt.model_t == 'resnet50':
                feat_t, logit_t = model_t(input.cuda(), is_feat=True)
            else:
                feat_t, logit_t = model_t(input.cuda(), is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t] 

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        
        if opt.TDA == 'True':
            if opt.tda_layer_t == 0 :
                f_s_tda = logit_s
                f_t_tda = logit_t
            else :
                f_s_tda = feat_s[opt.tda_layer_s]
                f_t_tda = feat_t[opt.tda_layer_t]
                if opt.tda_layer_s != -1:
                    if rips_pool_s is not None:
                        f_s_tda = rips_pool_s(f_s_tda)
                    else :
                        f_s_tda = rips_pool_t(f_s_tda)
                if opt.tda_layer_t != -1:
                    f_t_tda = rips_pool_t(f_t_tda)
                if model_t.nChannels[opt.tda_layer_t] != model_s.nChannels[opt.tda_layer_s]:
                    f_s_tda = rips_1dconv(f_s_tda)
            
            loss_tda = criterion_tda(ripsnet_t(f_t_tda.reshape(f_t_tda.shape[0], -1)), ripsnet_t(f_s_tda.reshape(f_s_tda.shape[0], -1)))  


        # other kd beyond KL divergence
        if opt.distill == 'KD':
            loss_kd = 0
        elif opt.distill == 'hint_L2' or 'hint_KD':
            f_s = (feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        else :
            raise NotImplementedError(opt.distill)
                
        if opt.TDA == 'True':
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd + opt.delta * loss_tda
        else :        
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            if opt.TDA == 'True' :
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Cls loss {3:.4f}\t'
                'Div loss {4:.4f}\t'
                'KD loss {5:.4f}\t'
                'TDA loss {6:.4f}\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), loss_cls, loss_div, loss_kd, loss_tda, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
                
                wandb.log({"train": {"time": batch_time.avg, "data_time": data_time.avg, "loss": losses.avg, "cls loss": loss_cls, "div loss": loss_div, "kd loss": loss_kd, "tda loss": loss_tda, "acc": top1.avg}})

            else :
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Cls loss {3:.4f}\t'
                    'Div loss {4:.4f}\t'
                    'KD loss {5:.4f}\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, idx, len(train_loader), loss_cls, loss_div, loss_kd, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()


    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                wandb.log({"val": {"time": batch_time.avg, "loss": losses.avg, "acc": top1.avg}})

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg