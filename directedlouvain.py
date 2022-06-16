def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, importlib
    __file__ = pkg_resources.resource_filename(__name__,'directedlouvain.so')
    __loader__ = None; del __bootstrap__, __loader__
    importlib.import_module(__name__,__file__)
    # importlib.load_dynamic(__name__,__file__)


__bootstrap__()