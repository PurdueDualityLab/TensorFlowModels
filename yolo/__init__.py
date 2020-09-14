import importlib as _imp
import sys as _sys

_dict = {
    'DarkNet53': 'yolo.modeling.yolo_v3',
    'Yolov3': 'yolo.modeling.yolo_v3',
    'Yolov4': 'yolo.modeling.yolo_v4',
}

__all__ = list(_dict)

if _sys.version_info < (3, 7):
    from yolo.modeling.yolo_v3 import DarkNet53
    from yolo.modeling.yolo_v3 import Yolov3
    from yolo.modeling.yolo_v4 import Yolov4
else:
    def __getattr__(name):
        module = _dict[name]
        module = _imp.import_module(module)
        return getattr(module, name)
    def __dir__():
        return [*_sys.modules[__name__].__dict__, *_dict]
