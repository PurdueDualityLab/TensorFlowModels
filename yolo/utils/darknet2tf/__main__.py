#!/usr/bin/env python3
"Convert a DarkNet config file and weights into a TensorFlow model"

from absl import app
import sys
from . import main, _parser

if __name__ == '__main__':
    # I dislike Abseil's current help menu. I like the default Python one
    # better
    if '-h' in sys.argv or '--help' in sys.argv:
        _parser.parse_args(sys.argv[1:])
        exit()
    app.run(main)
