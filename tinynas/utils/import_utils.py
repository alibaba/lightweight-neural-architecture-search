import importlib.util
import os

def load_py_module_from_path(module_path, module_name=None):
    if module_path.find(':') > 0:
        split_path = module_path.split(':')
        module_path = split_path[0]
        function_name = split_path[1]
    else:
        function_name = None

    if module_name is None:
        module_name = module_path.replace('/', '_').replace('.', '_')

    assert os.path.isfile(module_path)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    any_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(any_module)
    if function_name is None:
        return any_module
    else:
        return getattr(any_module, function_name)
