# Python equivalent of pylint.sh script

""" Fail the program upon the first error, return first error code;
Python basically does this inherently 
    $ set -euo pipefail 
"""

""" Download latest configs from main TensorFlow repo
wget: downloads file from online
-q : quiet (no output)
-O : write documents to file path/name (output)
NOTE: "/tmp/" is for Linux; need to first determine OS, then which folders correspond to that 
    $ wget -q -O /tmp/pylintrc https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc
    $ wget -q -O /tmp/pylint_allowlist https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylint_allowlist
"""

# Configure it later on based on OS... 

import datetime
#import pylint
import os
import subprocess
import sys
import tempfile
from urllib import request

from pylint import epylint as lint

SCRIPT_DIR= tempfile.gettempdir()
PYLINTRC_DIR = os.path.join(tempfile.gettempdir(), "pylintrc")
PYLINT_ALLOWLIST_DIR = os.path.join(tempfile.gettempdir(), "pylint_allowlist")

# Download latest configs from main TensorFlow repo
request.urlretrieve("https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylintrc", PYLINTRC_DIR)
request.urlretrieve("https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/tools/ci_build/pylint_allowlist")

# Get the number of CPUs
def num_cpus():
    N_CPUS = os.cpu_count()
    if (N_CPUS is None):
        print("ERROR: Unable to determine the number of CPUs")
        exit()
    return N_CPUS

# Get list of all Python files, regardless of mode
def get_py_files_to_check():
    py_files_list = []
    current_dir = os.getcwd()
    for root, folders, files in os.walk(current_dir):
        for filename in folders + files:
            if filename.endswith(".py"):
                py_files_list.append(os.path.join(root, filename))
    return py_files_list

def do_pylint():
    # Something happened. TF no longer has Python code if this branch is taken
    PYTHON_SRC_FILES=get_py_files_to_check()
    if not PYTHON_SRC_FILES:
        print("do_pylint found no Python files to check. Returning.")
        exit()
    # Now that we know we have to do work, check if `pylint` is installed
    # if "pylint" in sys.modules:
    #     print("yes")
    # else:
    #     print("no")

    # Configure pylint using the following file
    PYLINTRC_FILE=PYLINTRC_DIR
    if not os.path.isfile(PYLINTRC_FILE):
        print("ERROR: Cannot find pylint rc file at "+PYLINTRC_FILE)
        exit()
    
    # Run pylint in parallel, after some disk setup
    NUM_SRC_FILES = len(PYTHON_SRC_FILES)
    NUM_CPUS = num_cpus()    
    print("Running pylint on %d files with %d parallel jobs...\n"
        %(NUM_SRC_FILES, NUM_CPUS))

    PYLINT_START_TIME = datetime.datetime.now()

    OUTPUT_FILE= tempfile.mkstemp(suffix="_pylint_output.log") 
    ERRORS_FILE= tempfile.mkstemp(suffix="_pylint_errors.log")
    PERMIT_FILE= tempfile.mkstemp(suffix="_pylint_permit.log")
    FORBID_FILE= tempfile.mkstemp(suffix="_pylint_forbid.log")

    # When running, filter to only contain the error code lines. Removes module
    # header, removes lines of context that show up from some lines.
    # Also, don't redirect stderr as this would hide pylint fatal errors.

    #with open("test.txt", "w") as f:

    #(pylint_stdout, pylint_stderr) = lint.py_run(PYTHON_SRC_FILES[0], return_std=True)
    out = subprocess.Popen(['pylint', '--rcfile='+PYLINTRC_FILE, '--output-format=parseable',
        '--jobs='+str(NUM_CPUS), PYTHON_SRC_FILES[0]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    stdout,stderr = out.communicate()
    print(stdout)
    
    #  ${PYLINT_BIN} --rcfile="${PYLINTRC_FILE}" --output-format=parseable \
    #  --jobs=${NUM_CPUS} ${PYTHON_SRC_FILES} | grep '\[[CEFW]' > ${OUTPUT_FILE}
 
# Close the file


do_pylint()

    
