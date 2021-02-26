import os
import glob

dir = os.path.dirname(__file__)
dest = os.path.join(dir, 'Kconfig')
template = os.path.join(dir, 'Kconfig.template')
creds = os.path.join(dir, 'creds', '*.json')

AUTH_KEY = next(glob.iglob(creds))

with open(template) as tfile, open(dest, 'w') as dfile:
  dfile.write(tfile.read().format(AUTH_KEY=repr(AUTH_KEY)))
