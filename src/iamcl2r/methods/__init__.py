import importlib
from iamcl2r.methods.hoc import HocLoss


__factory = {
    'er':           'iamcl2r.methods.er.ERconfigs',
    'hoc':          'iamcl2r.methods.hoc.HOCconfigs',
}


def wrapper_factory(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown method: {}".format(name))
    module_path = '.'.join(__factory[name].split('.')[:-1])
    module = importlib.import_module(module_path)
    class_name = __factory[name].split('.')[-1]
    return getattr(module, class_name)(**kwargs)


def set_method_configs(args, name='er'):
    method_configs = wrapper_factory(name)
    for k, v in method_configs.items():
        args.__setattr__(k, v)
