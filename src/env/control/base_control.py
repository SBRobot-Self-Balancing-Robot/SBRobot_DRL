import typing as T

class BaseControl:
    def __init__(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs) -> T.Any:
        raise NotImplementedError

    def step(self, *args, **kwargs) -> T.Any:
        raise NotImplementedError
    
    def update(self, *args, **kwargs) -> T.Any:
        raise NotImplementedError
    
    