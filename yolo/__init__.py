import importlib as _imp
import sys as _sys

_dict = {
    'DarkNet53': 'yolo.modeling.yolo_v3',
    'Yolov3': 'yolo.modeling.yolo_v3',
    'Yolov4': 'yolo.modeling.yolo_v4',
}

__all__ = list(_dict)

if _sys.version_info < (3, 7):
    from types import ModuleType as _ModuleType
    class _Module(_ModuleType):
        def __getattr__(self, name):
            module = _dict[name]
            module = _imp.import_module(module)
            return getattr(module, name)
        def __dir__(self):
            return [*_sys.modules[__name__].__dict__, *_dict]
    _sys.modules[__name__].__class__ = _Module
else:
    def __getattr__(name):
        module = _dict[name]
        module = _imp.import_module(module)
        return getattr(module, name)
    def __dir__():
        return [*_sys.modules[__name__].__dict__, *_dict]
