from . import config_classes as _conf
import collections.abc


class _DarkNetSectionList(collections.abc.MutableSequence):
    __slots__ = ['data']

    def __init__(self, initlist=None):
        self.data = []
        if initlist is not None:
            self.data = list(initlist)

    # Overriding Python list operations
    def __len__(self):
        return max(0, len(self.data) - 1)
    def __getitem__(self, i):
        if i >= 0:
            i += 1
        if isinstance(i, slice):
            return self.__class__(self.data[i])
        else:
            return self.data[i]
    def __setitem__(self, i, item):
        if i >= 0:
            i += 1
        self.data[i] = item
    def __delitem__(self, i):
        if i >= 0:
            i += 1
        del self.data[i]
    def insert(self, i, item):
        if i >= 0:
            i += 1
        self.data.insert(i, item)


class DarkNetModel(_DarkNetSectionList):
    """
    This is a special list-like object to handle the storage of layers in a
    model that is defined in the DarkNet format. Note that indexing layers in a
    DarkNet model can be unintuitive and doesn't follow the same conventions
    as a Python list.

    In DarkNet, a [net] section is at the top of every model definition. This
    section defines the input and training parameters for the entire model.
    As such, it is not a layer and cannot be referenced directly. For our
    convenience, we allowed relative references to [net] but disallowed absolute
    ones. Like the DarkNet implementation, our implementation numbers the first
    layer (after [net]) with a 0 and

    To use conventional list operations on the DarkNetModel object, use the data
    property provided by this class.
    """
    def to_tf(self):
        from yolo.modeling.building_blocks import YoloLayer
        tensors = _DarkNetSectionList()
        yolo_tensors = []
        for cfg in self.data:
            tensor = cfg.to_tf(tensors)
            assert tensor.shape[1:] == cfg.shape, str(cfg) + f" shape inconsistent\n\tExpected: {cfg.shape}\n\tGot: {tensor.shape[1:]}"
            if cfg._type == 'yolo':
                yolo_tensors.append((cfg, tensor))
            tensors.append(tensor)

        masks = {}
        anchors = None
        thresh = None
        class_thresh = None

        #yolo_layer = YoloLayer()
        return yolo_tensors

    @property
    def net(self):
        return self.data[0]
