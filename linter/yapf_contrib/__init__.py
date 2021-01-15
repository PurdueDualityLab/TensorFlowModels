# Enable yapf_contrib
# From https://github.com/google/yapf/tree/fixers/contrib

from yapf.yapflib import yapf_api
from yapf_contrib.fixers import fixers_api

enabled_fixers = []

old_FormatCode = yapf_api.FormatCode


def FormatCode(unformatted_source, *args, **kwargs):
  global enabled_fixers
  options = {'fixers': enabled_fixers}
  unformatted_source = fixers_api.Pre2to3FixerRun(unformatted_source, options)
  return old_FormatCode(unformatted_source, *args, **kwargs)


yapf_api.FormatCode = FormatCode
