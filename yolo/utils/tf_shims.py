"""
This file contains shims that allow you to access functions in future versions
of TensorFlow even on older versions.
"""

from tensorflow import keras as ks


def ks_Model___updated_config(self):
    '''
    You can use this shim to update an internal TensorFlow function to fix their
    bug when saving a model. When using this shim, you can serialize models that
    are in their own namespace.

    Example:
    @ks.utils.register_keras_serializable(package='yolo')
    class DarkNet53(ks.Model):
        """The Darknet Image Classification Network Using Darknet53 Backbone"""
        _updated_config = tf_shims.ks_Model___updated_config

    Applies to: all versions of TensorFlow
    '''
    model_config = ks.Model._updated_config(self)
    model_config['class_name'] = ks.utils.get_registered_name(self.__class__)
    return model_config
