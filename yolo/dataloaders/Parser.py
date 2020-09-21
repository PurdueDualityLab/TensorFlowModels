import abc
from typing import Callable

class DatasetParser(abc.ABC):
    @abc.abstractmethod
    def unbatched_process_fn(self, is_training: bool) -> Callable[[dict], dict]:
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
