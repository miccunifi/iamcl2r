import os.path as osp
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentParams(): 
    
    root_folder: str = './data'
    train_dataset_name: str = 'cifar'
    backbone: str = 'resnet18'
    data_path: str = './data'
    output_folder: str = './models_ckpts'

    # methods settings
    fixed: bool = False
    preallocated_classes: int = -1
    create_old_model: bool = False

    # iamcl2r settings
    pretrained_model_path: List[str] = field(default_factory=lambda: [])
    replace_ids: List[int] = field(default_factory=lambda: [])
    feat_size: int = 512
    use_embedding_layer: bool = False
    use_subsampled_dataset: bool = True
    img_per_class: int = 300
    replace_model_architecture: bool = False

    # training settings
    amp: bool = True
    train_only: bool = False
    epochs: int = 70
    batch_size: int = 128
    num_workers: int = 8
    initial_increment: int = 10
    increment: int = 15
    rehearsal: int = 20
    lr: float = 0.001
    min_lr: float = 0.00001
    weight_decay: float = 0.0005
    eval_period: int = 5
    save_period: int = 70
    save_best: bool = False  # if false save last, if true save best based on classification accuracy

    # reproducibility settings
    seed: int = 42

    # only eval settings
    eval_only: bool = False
    resume_path: str = ''
    number_training_classes: int = 100
    ntasks_eval: int = 1

    # distributed settings
    distributed: bool = True
    dist_url: str = 'env://'
    dist_backend: str = 'nccl'
    use_bn_sync: bool = True
    no_set_device_rank: bool = False
    ddp_static_graph: bool = False
    