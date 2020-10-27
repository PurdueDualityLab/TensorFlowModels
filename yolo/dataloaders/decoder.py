import abc


class Decoder(object):
    """Decodes the raw data into tensors."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def decode(self, serialized_example):
        """Decodes the serialized example into tensors.
    Args:
      serialized_example: a serialized string tensor that encodes the data.
    Returns:
      decoded_tensors: a dict of Tensors.
    """
        pass
