import torch
import torch.nn as nn 

from collections import OrderedDict

from iamcl2r.utils import l2_norm
from iamcl2r.models.resnet import resnet18, resnet50
from iamcl2r.models.senet import SENet18
from iamcl2r.models.regnet import RegNetY_400MF

import logging
logger = logging.getLogger('Model')


__BACKBONE_OUT_DIM = {
    'resnet18': 512,
    'senet18': 512,
    'regnet400': 384,
}


def get_backbone_feat_size(backbone):
    if backbone not in __BACKBONE_OUT_DIM:
        raise ValueError('Backbone not supported: {}'.format(backbone))
    return __BACKBONE_OUT_DIM[backbone]


def extract_features(args, device, net, loader, return_labels=False):
    features = None
    labels = None
    net.eval()
    with torch.no_grad():
        for inputs in loader:
            images = inputs[0].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                f = net(images)['features']
            f = l2_norm(f)
            if features is not None:
                features = torch.cat((features, f), 0)
                labels = torch.cat((labels, inputs[1]), 0) if return_labels else None
            else:
                features = f
                labels = inputs[1] if return_labels else None
    if return_labels:
        return features.detach().cpu(), labels.detach().cpu()
    return features.detach().cpu().numpy()


class Incremental_ResNet(nn.Module):
    def __init__(self, 
                 num_classes=100, 
                 feat_size=99, 
                 backbone='resnet18', 
                ):
        
        super(Incremental_ResNet, self).__init__()
        self.feat_size = feat_size
        
        if backbone == 'resnet18':
            self.backbone = resnet18()
        elif backbone == 'resnet50':
            self.backbone = resnet50()
        elif backbone == 'senet18':
            self.backbone = SENet18()
        elif backbone == 'regnet400':
            self.backbone = RegNetY_400MF()
        else:
            raise ValueError('Backbone not supported: {}'.format(backbone))

        self.out_dim = self.backbone.out_dim
        self.feat_size = feat_size
        
        self.fc1 = None 
        self.fc2 = None
        if self.out_dim != self.feat_size:
            logger.info(f"add a linear layer from {self.out_dim} to {self.feat_size}")
            self.fc1 = nn.Linear(self.out_dim, self.feat_size, bias=False)
        self.fc2 = nn.Linear(self.feat_size, num_classes, bias=False)  # classifier
        
            
    def forward(self, x, return_dict=True):
        x = self.backbone(x)

        if self.fc1 is not None:
            z = self.fc1(x)
        else:
            z = x
        
        y = self.fc2(z)

        if return_dict:
            return {'backbone_features': x,
                    'logits': y, 
                    'features': z
                    }

        else:
            return x, y, z
    
    def expand_classifier(self, new_classes):
        old_classes = self.fc2.weight.data.shape[0]
        old_weight = self.fc2.weight.data
        self.fc2 = nn.Linear(self.feat_size, old_classes + new_classes, bias=False)
        self.fc2.weight.data[:old_classes] = old_weight
        

def dsimplex(num_classes=100, device='cuda'):
    def simplex_coordinates_gpu(n, device):
        t = torch.zeros((n + 1, n), device=device)
        torch.eye(n, out=t[:-1,:], device=device)
        val = (1.0 - torch.sqrt(1.0 + torch.tensor([n], device=device))) / n
        t[-1,:].add_(val)
        t.add_(-torch.mean(t, dim=0))
        t.div_(torch.norm(t, p=2, dim=1, keepdim=True)+ 1e-8)
        return t
        
    feat_dim = num_classes - 1
    ds = simplex_coordinates_gpu(feat_dim, device)#.cpu()
    return ds


def create_model(args, 
                 device,
                 resume_path=None, 
                 num_classes=None, 
                 feat_size=None, 
                 backbone=None, 
                 new_classes=None,
                 **kwargs):

    if backbone is None:
        backbone = args.backbone
    if feat_size is None and not args.use_embedding_layer:
        if args.fixed:
            feat_size = args.preallocated_classes - 1
        else:
            feat_size = get_backbone_feat_size(backbone)
        args.feat_size = feat_size
    elif args.use_embedding_layer:
        assert args.feat_size is not None, "feat_size must be set in configs when using embedding layer"
        feat_size = args.feat_size
    else:
        raise ValueError('feat_size not set')

    if num_classes is None:
        num_classes = args.classes_at_task[args.current_task_id-1]
    if new_classes is None and args.current_task_id > 0 and not args.fixed:
        new_classes = len(args.new_data_ids_at_task[args.current_task_id])
    if args.fixed:
        num_classes = args.preallocated_classes

    logger.info(f"Creating model with {num_classes} classes and {feat_size} features")

    model_cfg = {
        'num_classes': num_classes,
        'feat_size': feat_size,
        'backbone': backbone,
    }
    model = Incremental_ResNet(**model_cfg)

    if args.fixed:
        fixed_weights = dsimplex(num_classes=num_classes, device=device)
        logger.info(f"Fixed weights shape: {fixed_weights.shape}")
        model.fc2.weight.requires_grad = False  # set no gradient for the fixed classifier
        model.fc2.weight.copy_(fixed_weights)   # set the weights for the classifier

    if resume_path not in [None, '']:
        logger.info(f"Resuming Weights from {resume_path}")
        new_pretrained_dict = torch.load(resume_path, map_location='cpu')
        if "net" in new_pretrained_dict.keys():
            new_pretrained_dict = new_pretrained_dict["net"]

        if "pretrained" in resume_path:
            state_dict = OrderedDict()
            for k, v in new_pretrained_dict.items():
                name = k.replace('.blocks.', '.')
                if name not in model.state_dict().keys():
                    logger.info(f"{name} \t not found!!!!!!")
                    continue
                state_dict[name] = v
            del state_dict['fc2.weight'] # remove classifier weights from iamcl2r pretrained weights
        else:
            state_dict = new_pretrained_dict

        model.load_state_dict(state_dict, strict=False)    # the dict does not have always the old_classifier weights
    
    if new_classes is not None and new_classes > 0 and not args.fixed:
        logger.info(f"Expanding classifier to {num_classes + new_classes} classes")
        model.expand_classifier(new_classes)
    
    model.to(device=device)
    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    return model
