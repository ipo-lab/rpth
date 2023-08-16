def rpth_control(max_iters=10, eps=1e-3, normalize=True, orthant=None, verbose=False, **kwargs):
    control = {"max_iters": max_iters,
               "eps": eps,
               "normalize": normalize,
               "orthant": orthant,
               "verbose": verbose
               }
    control.update(**kwargs)
    return control
