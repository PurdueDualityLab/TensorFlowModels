import abc
from typing import Callable

class Parser(object):
    """Parses data and produces tensors to be consumed by models."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _parse_train_data(self, decoded_tensors):
        """Generates images and labels that are usable for model training.
        Args:
            decoded_tensors: a dict of Tensors produced by the decoder.
        Returns:
            images: the image tensor.
            labels: a dict of Tensors that contains labels.
        """
        pass

    @abc.abstractmethod
    def _parse_eval_data(self, decoded_tensors):
        """Generates images and labels that are usable for model evaluation.
        Args:
            decoded_tensors: a dict of Tensors produced by the decoder.
        Returns:
            images: the image tensor.
            labels: a dict of Tensors that contains labels.
        """
        pass

    def parse_fn(self, is_training):
        """Returns a parse fn that reads and parses raw tensors from the decoder.
        Args:
            is_training: a `bool` to indicate whether it is in training mode.
        Returns:
            parse: a `callable` that takes the serialized examle and generate the
                images, labels tuple where labels is a dict of Tensors that contains
                labels.
        """
        def parse(decoded_tensors):
            """Parses the serialized example data."""
            if is_training:
                return self._parse_train_data(decoded_tensors)
            else:
                return self._parse_eval_data(decoded_tensors)

        return parse


class DatasetParser(abc.ABC):
    @abc.abstractmethod
    def unbatched_process_fn(self,
                             is_training: bool) -> Callable[[dict], dict]:
        """
        Create a function that decodes the dataset and returns a function
        that returns the image, bouding box, label, and any other required
        information needed for training or validation.

        Arguments:
            is_training: The data set will be formatted for training instead of
                         for validation.

        Returns: function to preprocess the data
        """
        ...

    @abc.abstractmethod
    def batched_process_fn(self, is_training: bool) -> Callable[[dict], dict]:
        """
        Apply normal preprocessing to the dataset to prepare it for training or
        validation. This includes jitter, resizing, and image manipulation.

        Arguments:
            is_training: The data set will be formatted for training instead of
                         for validation.

        Returns: function to preprocess the data
        """
        ...

    @abc.abstractmethod
    def build_gt(self, is_training: bool) -> Callable[[dict], dict]:
        """
        Format the ground truth so it can be used by loss functions.

        Arguments:
            is_training: The data set will be formatted for training instead of
                         for validation.

        Returns: function to preprocess the data
        """
        ...