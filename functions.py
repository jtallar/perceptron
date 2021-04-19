import numpy as np
import math


def get_activation_functions(name: str, beta: float, complex_error_enhance: bool, number_class):
    act_funcs_dict = {
        "sign": lambda x: np.asarray(np.sign(x), number_class),
        "linear": lambda x: x,
        "tanh": lambda x: np.tanh(x * beta),
        "exp": lambda x: 1 / (1 + math.e ** (-2 * beta * x)),
    }

    # the derived activation func, if it cant be, then return always 1
    act_funcs_der_dict = {
        "sign": lambda x: 1,
        "linear": lambda x: 1,
        "tanh": lambda x: beta * (1 - act_funcs_dict["tanh"](x) ** 2),
        "exp": lambda x: 2 * beta * act_funcs_dict["exp"](x) * (1 - act_funcs_dict["exp"](x))
    }

    # the derived activation func but with the new error calculus, if it cant be, then return always 1
    act_funcs_der_2_dict = {
        "sign": lambda x: 1,
        "linear": lambda x: 1,
        "tanh": lambda x: beta,
        "exp": lambda x: 2 * beta
    }

    # in case it is complex and a desired to make better error
    if complex_error_enhance:
        # TODO: maybe check if it is not tanh and exit
        return act_funcs_dict[name], act_funcs_der_2_dict[name]

    return act_funcs_dict[name], act_funcs_der_dict[name]
