import importlib as _imp
import sys as _sys

_dict = {
    'DarkNet53': 'yolo.modeling.yolo_v3',
    'Yolov3': 'yolo.modeling.yolo_v3',
    'Yolov4': 'yolo.modeling.yolo_v4',
}

__all__ = list(_dict)

if _sys.version_info < (3, 7):
    # Work around to use the __getattr__ in a module
    # https://stackoverflow.com/a/1463773
    class _Sneaky(object):
      def __init__(self):
        self.download = None

      @property
      def DarkNet53(self):
        from yolo.modeling.yolo_v3 import DarkNet53
        return DarkNet53

      @property
      def Yolov3(self):
        from yolo.modeling.yolo_v3 import Yolov3
        return Yolov3

      @property
      def Yolov4(self):
        from yolo.modeling.yolo_v4 import Yolov4
        return Yolov4

      def __getattr__(self, name):
        return globals()[name]

    _sys.modules[__name__] = _Sneaky()
else:
    def __getattr__(name):
        module = _dict[name]
        module = _imp.import_module(module)
        return getattr(module, name)
    def __dir__():
        return [*_sys.modules[__name__].__dict__, *_dict]
