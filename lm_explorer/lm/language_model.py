import functools

import torch

_lm_cls = {}


class LanguageModelMeta(type):
    def __init__(self, name, bases, kwds):
        super().__init__(name, bases, kwds)
        _lm_cls[kwds.get('name')] = self

    @functools.lru_cache()
    def __call__(self, *args, **kwargs):
        if self.name != 'base':
            return super().__call__(*args, **kwargs)

        fullname = args[0] if args else kwargs['fullname']
        model, paramset = fullname.split('/')
        return _lm_cls[model](paramset)


class LanguageModel(metaclass=LanguageModelMeta):
    name = 'base'

    def __init__(self, model_name: str, *args, **kwargs):
        raise NotImplementedError

    def predict(self, previous: str, next: str) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, index: int) -> str:
        raise NotImplementedError
