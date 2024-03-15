import os.path as osp
import importlib
from torch.utils.data import DataLoader
from torchvision import transforms
from continuum import ClassIncremental
from continuum.rehearsal import RehearsalMemory

from iamcl2r.dataset.dataset_utils import *


__factory = {
    'cifar':    'iamcl2r.dataset.cifar.load_cifar',
}


__factory_modalities = ['train', 
                        'identification'
                       ]


def create_data_and_transforms(args, mode="train", return_dict=True):

    if args.train_dataset_name not in __factory.keys():
        raise KeyError(f"Unknown dataset: {args.train_dataset_name}")
    
    if mode not in __factory_modalities:
        raise KeyError(f"Unknown modality: {mode}")
    
    if not osp.exists(args.data_path):
        raise ValueError(f"Data path does not exist: {args.data_path}")
    
    scenario_train = None
    scenario_val = None
    memory = None 
    query_loader = None
    gallery_loader = None
    target_transform = None

    if mode=="train":
        if args.fixed:
            # local classes are written from the bottom of the shared interface w the server
            target_transform = transforms.Lambda(lambda y: args.preallocated_classes - 1 - y)
            
        logger.info(f"Loading Datasets")        
        module_path = '.'.join(__factory[args.train_dataset_name].split('.')[:-1])
        module = importlib.import_module(module_path)
        class_name = __factory[args.train_dataset_name].split('.')[-1]
        
        data_kwargs = {"path": args.data_path,
                       "use_subsampled_dataset": args.use_subsampled_dataset,
                       "img_per_class": args.img_per_class,
                      }
        data = getattr(module, class_name)(**data_kwargs)
        args.train_transform = data["train_transform"]

        # create task-sets for sequential fine-tuning learning
        scenario_train = ClassIncremental(data["dataset_train"],
                                          initial_increment=args.initial_increment,
                                          increment=args.increment,
                                          transformations=data["train_transform"]
                                          ) 
        args.num_classes = scenario_train.nb_classes
        args.nb_tasks = scenario_train.nb_tasks
        args.nb_tasks_evaluation = scenario_train.nb_tasks 

        logger.info(f"\n\nTraining with {args.nb_tasks} tasks.\nIn the first task there are {args.initial_increment} classes, while the other tasks have {args.increment} classes each.\n\n")

        scenario_val = ClassIncremental(data["dataset_val"],
                                        initial_increment=args.initial_increment,
                                        increment=args.increment,
                                        transformations=data["val_transform"]
                                        ) 
        
        # create episodic memory dataset
        memory = RehearsalMemory(memory_size=args.num_classes * args.rehearsal,
                                 herding_method="random",
                                 fixed_memory=True,
                                 nb_total_classes=args.num_classes
                                )
        
        
        if return_dict:
            return {"scenario_train": scenario_train, 
                    "scenario_val": scenario_val, 
                    "memory": memory, 
                    "target_transform": target_transform
                    }
        return scenario_train, scenario_val, memory, target_transform

    else:        
        module_path = '.'.join(__factory[args.train_dataset_name].split('.')[:-1])
        module = importlib.import_module(module_path)
        class_name = __factory[args.train_dataset_name].split('.')[-1] + f'_{mode}'
        try:
            gallery_set, query_set = getattr(module, class_name)(path=args.data_path)
        except AttributeError:
            raise AttributeError(f"Please implement the function 'load_{args.train_dataset_name}_{mode}'")
        
        query_loader = DataLoader(query_set, batch_size=args.batch_size, 
                                shuffle=False, drop_last=False, 
                                num_workers=args.num_workers)
        gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size,
                                    shuffle=False, drop_last=False, 
                                    num_workers=args.num_workers)
        
        if return_dict:
            return {"query_loader": query_loader, 
                    "gallery_loader": gallery_loader, 
                    }
        return query_loader, gallery_loader, target_transform
