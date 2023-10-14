import inspect
import pygrasph
import numpy as np

def get_kernels():
    all_clases = []
    for name, cls in inspect.getmembers(pygrasph.kernels, inspect.isclass):
        if name != "_template_kernel":
            all_clases.append(cls)
    return all_clases

def init_kernel(knl):
    "Helper"
    return knl(1., 1.)

def test_kernel_attributes():
    kernel_classes = get_kernels()
    for cls in kernel_classes:
        try:
            kernel = init_kernel(cls)
            h = kernel.h
            k = kernel.k
        except TypeError:
            assert False, f"Failed to intialize {cls.__name__}."
        except AttributeError:
            assert False, f"{cls.__name__} doesn't have h or k attributes."

    assert True

def test_kernel_methods():
    kernel_classes = get_kernels()
    for cls in kernel_classes:
        kernel = init_kernel(cls)
        try:
            w = kernel.w(np.array([0.5]))
            dwdx = kernel.dwdx(np.array([[0.5, 0.5]]))
        except AttributeError:
            assert False, f"{cls.__name__} doesn't have w or dwdx functions."
        except TypeError:
            assert False, f"{cls.__name__} w or dwdx functions don't accept the correct inputs."

    assert True