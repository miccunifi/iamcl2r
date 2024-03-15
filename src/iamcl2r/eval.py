import numpy as np
import os.path as osp
import wandb

from iamcl2r.models import create_model, get_backbone_feat_size, extract_features
from iamcl2r.compatibility_metrics import average_compatibility, average_accuracy
from iamcl2r.performance_metrics import identification

import logging
logger = logging.getLogger('Eval')


def evaluate(args, device, query_loader, gallery_loader, ntasks_eval=None):

    if ntasks_eval is None: 
        ntasks_eval = args.ntasks_eval
    compatibility_matrix = np.zeros((ntasks_eval, ntasks_eval))
    targets = query_loader.dataset.targets
    gallery_targets = gallery_loader.dataset.targets

    for task_id in range(ntasks_eval):
        ckpt_path = osp.join(*(args.checkpoint_path, f"ckpt_{task_id}.pt")) 
        if not osp.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist. All the checkpoints need to have the format 'ckpt_<id>.pt' where id is the task id.")
        if args.fixed: 
            num_classes = args.preallocated_classes
        else:
            num_classes = args.classes_at_task[task_id]
        if args.replace_model_architecture: 
            raise NotImplementedError("Change model arch not implemented.")
        else:
            backbone_new = args.backbone
        logger.info(f"backbone: {backbone_new}")
        net = create_model(args,
                           device,
                           resume_path=ckpt_path, 
                           num_classes=num_classes, 
                           backbone=backbone_new,
                           new_classes=0,
                          )
        net.eval() 

        query_feat = extract_features(args, device, net, query_loader)

        for i in range(task_id+1):
            ckpt_path = osp.join(*(args.checkpoint_path, f"ckpt_{i}.pt")) 
            if not osp.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist. All the checkpoints need to have the format 'ckpt_<id>.pt' where id is the task id.")
            if args.fixed:
                num_classes = args.preallocated_classes
            else:
                num_classes = args.classes_at_task[i]
            if args.replace_model_architecture:
                raise NotImplementedError("Change model arch not implemented.")
            else:
                backbone = args.backbone
            logger.info(f"backbone: {backbone}")
            previous_net = create_model(args,
                                        device,
                                        resume_path=ckpt_path, 
                                        num_classes=num_classes, 
                                        backbone=backbone,
                                        new_classes=0,
                                        )
            previous_net.eval() 
            
            gallery_feat = extract_features(args, device, previous_net, gallery_loader)

            acc = identification(gallery_feat, gallery_targets, 
                                 query_feat, targets, 
                                 topk=1
                                )

            compatibility_matrix[task_id][i] = acc
            if i != task_id:
                acc_str = f'Cross-test accuracy between model at task {task_id+1} and {i+1}:'
            else:
                acc_str = f'Self-test of model at task {i+1}:'
            acc_str += f' 1:N search acc: {acc:.2f}'
            logger.info(f'{acc_str}')
        
    logger.info(f"Compatibility Matrix:\n{compatibility_matrix}")

    if compatibility_matrix.shape[0] > 1:
        # compatibility metrics
        ac = average_compatibility(matrix=compatibility_matrix)
        am = average_accuracy(matrix=compatibility_matrix)

        logger.info(f"Avg. Comp. = {ac:.2f}")
        logger.info(f"AM. Comp. = {am:.3f}")

        if args.is_main_process:
            wandb.log({f"eval/comp-acc": ac, 
                    f"eval/comp-am": am,
                    })

        # create a txt file with the compatibility matrix printed
        with open(osp.join(*(args.checkpoint_path, f'comp-matrix.txt')), 'w') as f:
            f.write(f"Compatibility Matrix ID:\n{compatibility_matrix}\n")
            f.write(f"Avg. Comp. = {ac:.2f}\n")
            f.write(f"AM. Comp. = {am:.3f}\n")


def validation(args, device, net, query_loader, gallery_loader, task_id, selftest=False):
    targets = query_loader.dataset.targets
    gallery_targets = gallery_loader.dataset.targets
        
    net.eval() 
    query_feat = extract_features(args, device, net, query_loader)
    
    if selftest:
        previous_net = net
    else:
        ckpt_path_val = osp.join(*(args.checkpoint_path, f"ckpt_{task_id-1}.pt")) 
        if args.fixed and not args.maximum_class_separation: 
            num_classes = args.preallocated_classes
        else:
            num_classes = args.classes_at_task[task_id-1]
        if args.replace_model_architecture:
            raise NotImplementedError("Change model arch not implemented in evaluation")
        else:
            backbone = args.backbone
        logger.info(f"backbone: {backbone}")
        previous_net = create_model(args,
                                    resume_path=ckpt_path_val, 
                                    num_classes=num_classes, 
                                    backbone=backbone,
                                    )
        previous_net.eval() 
        previous_net.to(device)
    
    gallery_feat = extract_features(args, device, previous_net, gallery_loader)
    acc = identification(gallery_feat, gallery_targets, 
                         query_feat, targets, 
                         topk=1)
    logger.info(f"{'Self' if selftest else 'Cross'} 1:N search Accuracy: {acc*100:.2f}")
    return acc


