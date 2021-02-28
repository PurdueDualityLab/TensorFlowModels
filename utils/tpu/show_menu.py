#!/usr/bin/env python3
import ast
import configparser
import contextlib
import curses
import json
from distutils.util import strtobool
import subprocess
import sys
import os

USE_GUI = False
DRY_RUN = True

if USE_GUI:
  import guiconfig
  menuconfig = guiconfig
else:
  import menuconfig
import kconfiglib

#menuconfig.menuconfig(kconfiglib.standard_kconfig(''))

kconf = kconfiglib.standard_kconfig('')
status = ''

def _quit(msg=None):
  global status
  guiconfig._root.destroy()
  if msg:
    status = msg

if USE_GUI:
  guiconfig._quit = _quit
  guiconfig.menuconfig(kconf)
else:
  with contextlib.redirect_stdout(sys.stderr):
    menuconfig._kconf = kconf
    menuconfig._conf_filename = menuconfig.standard_config_filename()
    menuconfig._conf_changed = menuconfig._load_config()
    menuconfig._minconf_filename = 'config'
    menuconfig._show_all = False
    kconf.warn = False
    os.environ.setdefault("ESCDELAY", "0")
  status = curses.wrapper(menuconfig._menuconfig)

if 'was not saved' in status:
  print('Changes not saved')
  exit(1)

if not menuconfig._conf_changed:
  if input('Spin up the same instance as last time? [yN] ').lower() != 'y':
    print('No new instances')
    exit(1)

my_env = {}
parser = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
def defaultSect(fp): yield '[DEFAULT]\n'; yield from fp
settingsFile = '.config'
if os.path.isfile(settingsFile):
    with open(settingsFile) as stream:
        parser.read_file(defaultSect(stream))
        for k, v in parser["DEFAULT"].items():
            my_env.setdefault(k.upper(), v)


CPU = strtobool(my_env.get('CONFIG_VM', 'n'))
TPU = strtobool(my_env.get('CONFIG_TPU', 'n'))
AUTH_KEY = ast.literal_eval(my_env['CONFIG_AUTH_KEY'])
VM_NAME = ast.literal_eval(my_env.get('CONFIG_VM_NAME', '""'))
TPU_NAME = ast.literal_eval(my_env.get('CONFIG_TPU_NAME', '""'))

if my_env.get('CONFIG_TPU_ZONE__US_CENTRAL1_F', 'n') == 'y':
  TPU_ZONE = "us-central1-f"
elif my_env.get('CONFIG_TPU_ZONE__US_CENTRAL1_A', 'n') == 'y':
  TPU_ZONE = "us-central1-a"
elif my_env.get('CONFIG_TPU_ZONE__EUROPE_WEST4_A', 'n') == 'y':
  TPU_ZONE = "europe-west4-a"
else:
  raise ValueError("Unknown Zone")

if CPU:
  if my_env.get('CONFIG_CPU_SIZE__E2_STANDARD_8', 'n') == 'y':
    CPU_SIZE = "e2-standard-8"
  elif my_env.get('CONFIG_CPU_SIZE__E2_STANDARD_16', 'n') == 'y':
    CPU_SIZE = "e2-standard-16"
  else:
    raise ValueError("Unknown CPU")

if TPU:
  if my_env.get('CONFIG_TPU_SIZE__V2_256', 'n') == 'y':
    TPU_SIZE = "v2-256"
  elif my_env.get('CONFIG_TPU_SIZE__V2_8', 'n') == 'y':
    TPU_SIZE = "v2-8"
  elif my_env.get('CONFITPU_SIZE__V3_8', 'n') == 'y':
    TPU_SIZE = "v3-8"
  else:
    raise ValueError("Unknown TPU")


if my_env.get('TF_VERSION__2_4_1', 'n') == 'y':
  TF_VERSION = '2.4.1'
else:
  raise ValueError("Unknown TensorFlow version")


with open(AUTH_KEY) as file:
  PROJECT_NAME = json.load(file)['project_id']

def new_cpu_tpu():
  command = ['ctpu', '--require-permissions', 'up', f'--tpu-size={TPU_SIZE}', f'--name={VM_NAME}', f'--project={PROJECT_NAME}', f'--zone={TPU_ZONE}', '--disk-size-gb=50', f'--machine-type={CPU_SIZE}', f'--tf-version={TF_VERSION}']
  if DRY_RUN:
    command.insert(0, 'echo')
  proc = subprocess.Popen(command, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
  proc.communicate()

def new_cpu():
  command = ['ctpu', '--require-permissions', 'up', '--vm-only', f'--name={VM_NAME}', f'--project={PROJECT_NAME}', f'--zone={TPU_ZONE}', '--disk-size-gb=50', f'--machine-type={CPU_SIZE}', f'--tf-version={TF_VERSION}']
  if DRY_RUN:
    command.insert(0, 'echo')
  proc = subprocess.Popen(command, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
  proc.communicate()

def new_tpu():
  command = ['ctpu', '--require-permissions', 'up', '--tpu-only', f'--tpu-size={TPU_SIZE}', f'--name={TPU_NAME}', f'--project={PROJECT_NAME}', f'--zone={TPU_ZONE}', f'--tf-version={TF_VERSION}']
  if DRY_RUN:
    command.insert(0, 'echo')
  proc = subprocess.Popen(command, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
  proc.communicate()

if CPU:
  if TPU:
    if TPU_NAME == VM_NAME:
      new_cpu_tpu()
    else:
      new_cpu()
      new_tpu()
  else:
    new_cpu()
else:
  if TPU:
    new_tpu()
  else:
    print("Nothing to configure")
    exit(1)
