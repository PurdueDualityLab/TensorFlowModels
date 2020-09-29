from absl.testing import parameterized
import tensorflow as tf

from yolo.modeling import building_blocks
from yolo.modeling.backbones import configs

try:
    from importlib import resources as importlib_resources
except BaseException:
    # Shim for Python 3.6 and older
    import importlib_resources

try:
    import yaml
except ImportError:
    yaml = None

from ..dn_yaml import *

test_params = [(config, config)
               for config in importlib_resources.contents(configs)
               if config.endswith('.yml')]


class dn_yaml_test(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(*test_params)
    def test_load_darknet_from_yaml(self, filename):
        # If pyyaml is not installed, skip it
        if yaml is None:
            self.skipTest("pyyaml must be installed to use dn_yaml extensions")

        # Test to see if the YAML is parsed correctly
        with importlib_resources.open_text(configs, filename) as file:
            config = file.read()
        loaded_model = load_darknet_from_yaml(config)

        # Test to see if there were any unregistered models in the YAML
        new_config = loaded_model.to_yaml()
        new_loaded_model = tf.keras.models.model_from_yaml(new_config)


if __name__ == "__main__":
    tf.test.main()
