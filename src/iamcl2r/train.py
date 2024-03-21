import numpy as np
import torch

import time
import wandb

from iamcl2r.utils import AverageMeter, log_epoch, l2_norm


def train_one_epoch(args,
                    device, 
                    net, 
                    previous_net, 
                    train_loader, 
                    scaler,
                    optimizer,
                    epoch, 
                    criterion_cls, 
                    task_id, 
                    add_loss,
                    target_transform=None
                   ):
    start = time.time()
    
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    net.train()
    for bid, batchdata in enumerate(train_loader):
        
        inputs = batchdata[0].to(device, non_blocking=True) 
        targets = batchdata[1].to(device, non_blocking=True)  

        if args.fixed:
            assert target_transform is not None, "target_transform is None"
            # transform targets to write for the end of the feature vector
            targets = target_transform(targets)
                
        optimizer.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
            output = net(inputs)
            loss = criterion_cls(output["logits"], targets)
        
            if previous_net is not None:
                with torch.no_grad():
                    feature_old = previous_net(inputs)["features"]
                if args.method == "hoc":
                    norm_feature_old = l2_norm(feature_old)
                    norm_feature_new = l2_norm(output["features"])
                    loss_feat = add_loss(norm_feature_new, norm_feature_old, targets)
                    # Eq. 3 in the paper
                    loss = loss * args.lambda_ + (1 - args.lambda_) * loss_feat
                else:
                    raise NotImplementedError(f"Method {args.method} not implemented")
        
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), inputs.size(0))

        acc_training = accuracy(output["logits"], targets, topk=(1,))
        acc_meter.update(acc_training[0].item(), inputs.size(0))
    
    # log after epoch
    if args.is_main_process:
        wandb.log({'train/epoch': epoch})
        wandb.log({'train/train_loss': loss_meter.avg})
        wandb.log({'train/train_acc': acc_meter.avg})
        wandb.log({'train/lr': optimizer.param_groups[0]['lr']})

    end = time.time()
    log_epoch(args.epochs, loss_meter.avg, acc_meter.avg, epoch=epoch, task=task_id, time=end-start)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def classification(args, device, net, loader, criterion_cls, target_transform=None):
    classification_loss_meter = AverageMeter()
    classification_acc_meter = AverageMeter()
    
    net.eval()
    with torch.no_grad():
        for bid, batchdata in enumerate(loader):
        
            inputs = batchdata[0].to(device, non_blocking=True) 
            targets = batchdata[1].to(device, non_blocking=True) 
            if args.fixed:
                # transform targets to write for the end of the interface vector
                targets = target_transform(targets)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                output = net(inputs)
                loss = criterion_cls(output["logits"], targets)

            classification_acc = accuracy(output["logits"], targets, topk=(1,))
            
            classification_loss_meter.update(loss.item(), inputs.size(0))
            classification_acc_meter.update(classification_acc[0].item(), inputs.size(0))

    log_epoch(loss=classification_loss_meter.avg, acc=classification_acc_meter.avg, classification=True)

    classification_acc = classification_acc_meter.avg

    return classification_acc