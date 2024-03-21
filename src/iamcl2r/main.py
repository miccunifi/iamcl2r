import os, argparse, yaml, wandb, datetime, logging, random
import numpy as np
import os.path as osp
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from iamcl2r.params import ExperimentParams
from iamcl2r.logger import setup_logger
from iamcl2r.methods import set_method_configs, HocLoss
from iamcl2r.dataset import create_data_and_transforms, BalancedBatchSampler
from iamcl2r.models import create_model
from iamcl2r.utils import check_params, save_checkpoint, init_distributed_device, is_master, broadcast_object
from iamcl2r.train import train_one_epoch, classification
from iamcl2r.eval import evaluate


def main():
    # load params from the config file from yaml to dataclass
    parser = argparse.ArgumentParser(description='Official PyTorch Implementation of "Stationary Representations: Optimally Approximating Compatibility and Implications for Improved Model Replacements" CVPR24')
    parser.add_argument("-c", "--config_path",
                        help="path of the experiment yaml",
                        default=osp.join(os.getcwd(), "configs/hoc.yaml"), 
                        type=str)
    params = parser.parse_args()

    with open(params.config_path, 'r') as stream:
        loaded_params = yaml.safe_load(stream)
    args = ExperimentParams()
    for k, v in loaded_params.items():
        args.__setattr__(k, v)
    args.yaml_name = os.path.basename(params.config_path)
    # reproducibility
    args.seed = np.random.randint(0, 10000) if args.seed == 0 else args.seed

    device = init_distributed_device(args)
    args.is_main_process = is_master(args)
    
    if not osp.exists(args.data_path) and args.is_main_process:
        os.makedirs(args.data_path)

    if not args.eval_only:
        checkpoint_path = osp.join(*(args.output_folder, 
                                     f"{datetime.datetime.now().strftime('%Y%m%d')}",
                                     f"{args.method}-{args.train_dataset_name}-{datetime.datetime.now().strftime('%H%M%S')}"
                                    )
                                  )
        if args.distributed:
            checkpoint_path = broadcast_object(args, checkpoint_path)
        args.checkpoint_path = checkpoint_path
        if not osp.exists(args.checkpoint_path) and args.is_main_process:
            os.makedirs(args.checkpoint_path)

    log_file = f"train-{datetime.datetime.now().strftime('%H%M%S')}-gpu{device.index}.log" if not args.eval_only else f"eval.log"
    setup_logger(logfile=os.path.join(*(args.checkpoint_path, log_file)),
                console_log=args.is_main_process, 
                file_log=True, 
                log_level="INFO") 
    
    if args.is_main_process:
        run = wandb.init(dir=".",tags=[]) 
    if not args.eval_only:
        logger = logging.getLogger('IAM-CL2R-Train')  
    else:
        logger = logging.getLogger('IAM-CL2R-Eval')
        run.tags = run.tags + ("eval",)
 
    set_method_configs(args, name=args.method)
    
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.replace_ids += [0]    
    args.replace_ids.sort()     
    
    set_method_configs(args, name=args.method)
    
    check_params(args)

    if args.is_main_process:
        wandb.config.update(vars(args))

    logger.info(f"Current args:\n{vars(args)}")
    logger.info(f"data of this run is stored in this path: {args.checkpoint_path}")
    logger.info(f'Logging on device {device}.')
    
    args.classes_at_task = []
    args.new_data_ids_at_task = []
    args.seen_classes = []

    if not args.eval_only:

        data = create_data_and_transforms(args)
        scenario_train = data["scenario_train"]
        scenario_val = data["scenario_val"]
        memory = data["memory"]
        target_transform = data["target_transform"]

        add_loss = None
        if args.method == "hoc":
            add_loss = HocLoss(mu_=args.mu_)

        best_acc = 0
        logger.info(f"Starting Training")

        for task_id, (train_task_set, _) in enumerate(zip(scenario_train, scenario_val)):
            args.current_task_id = task_id
            
            if task_id in args.replace_ids:
                resume_path = args.pretrained_model_path[args.replace_ids.index(task_id)]
            else:
                resume_path = osp.join(*(args.checkpoint_path, f"ckpt_{task_id-1}.pt"))
            
            new_data_ids = train_task_set.get_classes()
            val_dataset = scenario_val[:task_id + 1]

            class_in_step = scenario_train[:task_id].nb_classes + len(new_data_ids) if task_id > 0 else train_task_set.nb_classes
            args.classes_at_task.append(class_in_step)
            args.new_data_ids_at_task.append(new_data_ids)
            
            logger.info(f"Task {task_id+1} new classes in task: {new_data_ids}")

            previous_net = None
            if args.create_old_model:
                previous_net = create_model(args,
                                            device=device,
                                            resume_path=resume_path, 
                                            new_classes=0   # not expanding classifier for old model
                                        )
                # set false to require grad for all parameters
                for param in previous_net.parameters():
                    param.requires_grad = False
                previous_net.eval() 
            
            net = create_model(args, 
                               device=device,
                               resume_path=resume_path
                              )
            logger.info(f"Created model from {resume_path}")

            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)
            named_parameters = list(net.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
            
            optimizer = optim.SGD(
                                [
                                    {"params": gain_or_bias_params, "weight_decay": 0.},
                                    {"params": rest_params, "weight_decay": args.weight_decay},
                                ],
                                lr=args.lr, 
                                momentum=0.9, 
                                )
            scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
            scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
            criterion_cls = nn.CrossEntropyLoss().to(device)

            batchsampler = None
            batch_size = args.batch_size
            if task_id > 0: 
                if args.rehearsal > 0:
                    mem_x, mem_y, mem_t = memory.get()
                    train_task_set.add_samples(mem_x, mem_y, mem_t)
                batchsampler = BalancedBatchSampler(train_task_set, n_classes=train_task_set.nb_classes, 
                                                    batch_size=args.batch_size, n_samples=len(train_task_set._x), 
                                                    seen_classes=args.seen_classes, rehearsal=args.rehearsal)
                train_loader = DataLoader(train_task_set, batch_sampler=batchsampler, num_workers=args.num_workers) 
            else:
                if (args.use_subsampled_dataset and args.img_per_class * args.classes_at_task[0] < args.batch_size):
                    batch_size = args.img_per_class * args.classes_at_task[0]
                    logger.info(f"Original batch size of {batch_size} is too high.")
                    logger.info(f"In current task there are {args.classes_at_task[0]} classes and the dataset has {args.img_per_class} img per class.")
                    logger.info(f"Setting batch to {batch_size} images per class")
                train_loader = DataLoader(train_task_set, 
                                          batch_size=batch_size, shuffle=True, 
                                          drop_last=True, num_workers=args.num_workers) 
                
            val_loader = DataLoader(val_dataset, 
                                    batch_size=batch_size, shuffle=False,
                                    drop_last=False, num_workers=args.num_workers)
                
            best_acc = 0
            logger.info(f"Starting Epoch Loop at task {task_id+1}/{scenario_train.nb_tasks}")
            init_epoch = 0
            for epoch in range(init_epoch, args.epochs):
                args.current_epoch = epoch
                train_one_epoch(args, 
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
                                target_transform=target_transform
                                )
                scheduler_lr.step()

                if (epoch + 1) % args.eval_period == 0 or (epoch + 1) == args.epochs:
                    acc_val = classification(args, 
                                             device,
                                             net, 
                                             val_loader, 
                                             criterion_cls,
                                             target_transform=target_transform
                                            )
                    if args.is_main_process:
                        wandb.log({'val/val_acc': acc_val}) 
                                        
                    if (acc_val >= best_acc and args.save_best) or ((epoch + 1) == args.epochs and not args.save_best):
                        best_acc = acc_val
                        if args.is_main_process:
                            wandb.log({'val/best_acc': best_acc}) 
                            save_checkpoint(args, net, 
                                            optimizer, best_acc, scheduler_lr, backup=False)

                if ((epoch + 1) % args.save_period == 0 or (epoch + 1) == args.epochs) and args.is_main_process:
                    save_checkpoint(args, net, 
                                    optimizer, best_acc, scheduler_lr, backup=True)
                
            ## after training in current task
            if args.rehearsal > 0:
                memory.add(*scenario_train[task_id].get_raw_samples(), z=None)
                args.seen_classes = torch.tensor(list(memory.seen_classes), device=device)
                if args.is_main_process:
                    # save the memory after new data is added
                    memory.save(path=osp.join(args.checkpoint_path, "memory.npz"))   
                    logger.info(f"Memory saved in {osp.join(args.checkpoint_path, 'memory.npz')}")
                    save_checkpoint(args, net, optimizer, 
                                    best_acc, scheduler_lr, backup=True)
    
            if args.distributed:
                dist.barrier()
            
    if not args.train_only:
        logger.info(f"Starting Evaluation")
        if args.eval_only:
            assert osp.exists(args.checkpoint_path), f"Checkpoint path {args.checkpoint_path} does not exist"
            valid_ckpts_name = [1 for i in range(args.ntasks_eval) if osp.exists(osp.join(args.checkpoint_path,(f"ckpt_{i}.pt")))]
            assert len(valid_ckpts_name) == args.ntasks_eval, f"Checkpoint path {args.checkpoint_path} does not have all the required checkpoints or valid name format (ckpt_<TASK_ID>.pt)"
            args.classes_at_task = [np.arange(0, (i+1)*(args.number_training_classes//args.ntasks_eval)) for i in range(args.ntasks_eval)]
        data = create_data_and_transforms(args, mode="identification")
        query_loader = data["query_loader"]
        gallery_loader = data["gallery_loader"]
        evaluate(args, device, query_loader, gallery_loader,
                 ntasks_eval=(args.ntasks_eval if args.eval_only else args.nb_tasks_evaluation)
                )
        
        if args.is_main_process:
            artifact = wandb.Artifact('compatibility-matrix', type='text')
            artifact.add_file(osp.join(*(args.checkpoint_path, 'comp-matrix.txt')))
            run.log_artifact(artifact)
            wandb.finish()

    return 0

if __name__ == '__main__':
    main()