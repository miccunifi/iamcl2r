from continuum.datasets import CIFAR100
from torchvision.datasets import CIFAR10 as CIFAR10_torch

from torchvision.transforms import transforms
from iamcl2r.dataset.dataset_utils import subsample_dataset


def load_cifar(path, 
               input_size=32, 
               use_subsampled_dataset=False, 
               img_per_class=None, 
               ):

    train_transform = [transforms.Resize((input_size, input_size)),
                       transforms.RandomCrop(input_size, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5071, 0.4867, 0.4408),
                                            (0.2675, 0.2565, 0.2761))
                       ]
    dataset_train = CIFAR100(data_path=path, train=True, download=True)

    val_transform = [transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                        (0.2675, 0.2565, 0.2761))
                    ]
    dataset_val = CIFAR100(data_path=path, train=False, download=True)

    if use_subsampled_dataset:
        assert img_per_class is not None
        print(f"Subsampling dataset to {img_per_class} images per class.")
        dataset_train = subsample_dataset(dataset_train, img_per_class)
        dataset_val = subsample_dataset(dataset_val, img_per_class)
    
    return {
            "dataset_train":dataset_train, 
            "dataset_val": dataset_val, 
            "train_transform": train_transform,
            "val_transform": val_transform
            }


def load_cifar_identification(path, 
                              input_size=32
                              ):

    transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                        (0.2675, 0.2565, 0.2761))
                                    ])

    gallery_set = CIFAR10_torch(root=path, 
                                train=False, 
                                download=True, 
                                transform=transform
                               )
    
    query_set = CIFAR10_torch(root=path, 
                              train=True, 
                              download=True, 
                              transform=transform
                             )    
    return gallery_set, query_set

