#!/usr/bin/env python3
"Number the blocks in a DarkNet config file"

from absl import app, flags
from absl.flags import FLAGS as args

flags.DEFINE_string('filename', 'yolov3.cfg', 'name of the config file')

def main(argv):
    filename = args.filename
    with open(filename) as file:
        i = 0
        for line in file:
            if line.startswith('[') and line != '[net]\n':
                print(f"{i:4d}|{line}", end='')
                i += 1
            else:
                print(f"    |{line}", end='')

if __name__ == '__main__':
    app.run(main)
