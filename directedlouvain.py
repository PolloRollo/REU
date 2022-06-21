import importlib.util


def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import pkg_resources, importlib
    __file__ = pkg_resources.resource_filename(__name__, 'directedlouvain.so')
    __loader__ = None; del __bootstrap__, __loader__
    importlib.import_module(__name__, __file__)
    # load_dynamic(__name__, __file__)


def load_dynamic(module, path):
    spec = importlib.util.spec_from_file_location(module, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


__bootstrap__()
