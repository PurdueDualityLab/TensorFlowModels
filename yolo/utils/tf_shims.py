"""
This file contains shims that allow you to access functions in future versions
of TensorFlow even on older versions.
"""

import inspect
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
    model_config = _ks_Model___updated_config__old(self)
    model_config['class_name'] = ks.utils.get_registered_name(self.__class__)
    return model_config
_ks_Model___updated_config__old = ks.Model._updated_config

def ks_utils__register_keras_serializable(**kwargs):
    '''
    You can use this shim to ignore re-registering a model class when it is
    defined in __main__. This is needed for some Python libraries.

    Example:
    @tf_shims.ks_utils__register_keras_serializable(package='yolo')
    class DarkNet53(ks.Model):
        """The Darknet Image Classification Network Using Darknet53 Backbone"""

    Applies to: all versions of TensorFlow
    '''
    def do_nothing(**_kwargs):
        pass

    if inspect.getmodule(inspect.stack()[0][0]).__name__ != '__main__':
        return _ks_utils__register_keras_serializable__old(**kwargs)
    return do_nothing
_ks_utils__register_keras_serializable__old = ks.utils.register_keras_serializable

def _apply_globally():
    '''
    Apply the shims to any access of TensorFlow. This is probably not safe, but
    is convenient for testing.
    '''
    ks.Model._updated_config = _ks_Model___updated_config__old
    ks.utils.register_keras_serializable = _ks_utils__register_keras_serializable__old
