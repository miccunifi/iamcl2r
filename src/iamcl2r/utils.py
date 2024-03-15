import os
import os.path as osp
import numpy as np
import random

import torch
import torch.distributed as dist

import logging
logger = logging.getLogger('Utils')


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_epoch(n_epochs=None, loss=None, acc=None, epoch=None, task=None, time=None, classification=False):
    acc_str = f"Task {task + 1}" if task is not None else f""
    acc_str += f" Epoch [{epoch + 1}]/[{n_epochs}]" if epoch is not None else f""
    acc_str += f"\t Training Loss: {loss:.4f}" if loss is not None else f""
    acc_str += f"\t Training Accuracy: {acc:.2f}" if acc is not None else f""
    acc_str += f"\t Time: {time:.2f}" if time is not None else f""
    if classification:
        acc_str = acc_str.replace("Training", "Classification")   
    logger.info(acc_str)


def get_model_state_dict(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
        #todo consider to delete fc2 if args.fixed is true due to space occupation
    return model_state_dict


def save_checkpoint(args, 
                    net, 
                    optimizer, 
                    best_acc, 
                    scheduler_lr, 
                    backup=False, 
                    ):
    """ Save checkpoint. 
        If backup is True, the checkpoint is saved in the backup folder.
        Args:
            args: arguments
            net: model to save
            optimizer: optimizer to save
            scaler: scaler to save
            best_acc: best accuracy
            scheduler_lr: learning rate scheduler
            backup: if True, the checkpoint is saved in the backup folder
    """
    args.current_epoch += 1
    if args.current_epoch == args.epochs:
        args.current_epoch -= 2 
    ckpt = {'net': get_model_state_dict(net),
            'args': args,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'best_acc': best_acc,
            'scheduler_lr': scheduler_lr.state_dict() if scheduler_lr is not None else None,
           }
    ckpt_path = osp.join(args.checkpoint_path, f"{f'ckpt_{args.current_task_id}' if not backup else 'backup'}.pt")
    logger.info(f"Saving checkpoint in {ckpt_path}")
    torch.save(ckpt, ckpt_path)


def resume_rng_state_dict(ckpt):
    torch.set_rng_state(ckpt['cpu_rng_state'])
    torch.cuda.set_rng_state(ckpt['gpu_rng_state'])
    np.random.set_state(ckpt['numpy_rng_state'])
    random.setstate(ckpt['py_rng_state'])


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all([var in os.environ for var in pmi_vars]):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if is_using_distributed():
        # DDP via torchrun, torch.distributed.launch
        args.local_rank, _, _ = world_info_from_env()
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)
    return device


def broadcast_object(args, obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if args.rank == src:
        objects = [obj]
    else:
        objects = [None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def all_gather_object(args, obj, dst=0):
    # gather a pickle-able python object across all ranks
    objects = [None for _ in range(args.world_size)]
    dist.all_gather_object(objects, obj)
    return objects