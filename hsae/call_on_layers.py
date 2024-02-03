

def call_method_on_layers(
    function :Optional[Callable]=None, 
    preprocess=lambda s, *a, **k : (a, k),
    skip=lambda s, i, *a, **k: False
) -> Callable:
    """
    A decorator that calls functions with the same name on each layer of the 
    HSAE, adding the returned values to a list, which is passed to the wrapped
    function f as f(self, out, *args, **kwargs), where "out" is the list.

    Args:
        function (Optional[Callable]): The method to be called on each layer. Defaults to None.
        preprocess (Callable): A function that preprocesses the arguments before calling the method. 
            Defaults to lambda s, *a, **k : (a, k).
        skip (Callable): A function that determines whether to skip calling the method on a specific layer. 
            Defaults to lambda s, i, *a, **k: False.
            Currently, i == -1 corresponds to the top level SAE.

    Returns:
        Callable: A wrapped function that calls the given method on each layer.

    """
    funct = lambda s, *a, **k : function(s, *a, **k)
    if preprocess is True:
        return lambda f : (
            call_method_on_layers(
                function=f,
                preprocess=funct,
                skip=skip
            )
        )
    
    elif function is None:
        return lambda f : (
            call_method_on_layers(
                function=f,
                preprocess=preprocess,
                skip=skip
            )
        )

    function_name = function.__name__
    def wrapper(self, *args, **kwargs):
        args, kwargs = preprocess(self, *args, **kwargs)
        if skip(self, -1, *args, **kwargs):
            out = []
        else:
            out = [self.sae_0.__getattribute__(function_name)(*args, **kwargs)]
        for i, layers in enumerate(self.layers):
            if not skip(self, i, *args, **kwargs):
                out.append(layers.__getattribute__(function_name)(*args, **kwargs))
        return funct(self, out, *args, **kwargs)
    return wrapper
    




