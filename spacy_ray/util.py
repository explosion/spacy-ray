def set_params_proxy(model, proxy):
    """Set a 'proxy' on the internal ParamServer object for the model and
    its children. Experimental.
    """
    for node in model.walk():
        for name in node.param_names:
            if node.has_param(name):
                proxy.set_param(node.id, name, node.get_param(name))
        node._params.proxy = proxy
