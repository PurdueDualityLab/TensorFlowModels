import argparse
import glob
import os
import sys

import yapf
import yapf_contrib

OPTS = ['quotes']

parser = argparse.ArgumentParser(
    description="""Lint the repo and correct mistakes that are not caught by YAPF.

Options list:
- quotes:
    - bad-docstring-quotes (C0198)
    - inconsistent-quotes (W1405)
""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    prefix_chars='-+')
parser.add_argument(
    '-x',
    type=str,
    dest='noopts',
    metavar='OPTIONS',
    help='disable linting options')
parser.add_argument(
    '+x',
    type=str,
    dest='enopts',
    metavar='OPTIONS',
    help='enable linting options')
parser.add_argument(
    'files',
    nargs='+',
    help='if a directory is specified, all Python files inside are used')
args = parser.parse_args()

files = args.files
for i, file in enumerate(files):
  if os.path.isdir(file):
    files[i] = glob.glob(os.path.join(file, '**', '*.py'), recursive=True)
  elif not os.path.exists(file):
    raise FileNotFoundError(file)

opts = OPTS.copy()
if args.enopts is not None:
  remopts = args.enopts.split(',')
  for opt in remopts:
    opts.append(opt)
if args.noopts is not None:
  remopts = args.noopts.split(',')
  for opt in remopts:
    opts.remove(opt)

# The real linter starts here
if 'quotes' in opts:
  yapf_contrib.enabled_fixers.append('quotes')
yapf.main([sys.argv[0], '-i', *files])
