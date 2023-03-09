import distutils.dir_util
import pprint
import importlib.util

from .import_utils import load_py_module_from_path
from .dist_utils import master_only

import os

def mkfilepath(filename):
    filename = os.path.expanduser(filename)
    distutils.dir_util.mkpath(os.path.dirname(filename))


def mkdir(dirname):
    dirname = os.path.expanduser(dirname)
    distutils.dir_util.mkpath(dirname)


def robust_save(filename, save_function):
    mkfilepath(filename)
    backup_filename = filename + '.robust_save_temp'
    save_function(backup_filename)
    if os.path.isfile(filename):
        os.remove(filename)
    os.rename(backup_filename, filename)

@master_only 
def save_pyobj(filename, pyobj):
    mkfilepath(filename)
    the_s = pprint.pformat(pyobj, indent=2, width=120, compact=True)
    with open(filename, 'w') as fid:
        fid.write(the_s)

def load_pyobj(filename):
    with open(filename, 'r') as fid:
        the_s = fid.readlines()

    if isinstance(the_s, list):
        the_s = ''.join(the_s)

    the_s = the_s.replace('inf', '1e20')
    pyobj = ast.literal_eval(the_s)
    return pyobj
