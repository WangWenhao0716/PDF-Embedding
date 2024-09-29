from __future__ import absolute_import


from .vit_query import vit_base_query


__factory = {
    'vit_base_query': vit_base_query,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)
