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
            w = kernel.w(0.5)
            dwdx = kernel.dwdx(np.array([[0.5, 0.5]]))
        except AttributeError:
            assert False, f"{cls.__name__} doesn't have w or dwdx functions."
        except TypeError:
            assert False, f"{cls.__name__} w or dwdx functions don't accept the correct inputs."

    assert True

def test_wendlandc2_correctness():
     # known correct calculation
    dx = np.array([[0.5, 0.5]])
    r = np.linalg.norm(dx[0])
    h = 1.
    q = r/h
    alpha = 7./(64.*np.pi*h*h)
    w_ref = alpha*np.maximum(0., 2.-q)**4*(2.*q+1.)
    dwdx_ref = -alpha*10.*q*np.maximum(0., 2.-q)**3/(r*h)*dx

    # initialize kernel
    kernel = init_kernel(pygrasph.kernels.wendland_c2)

    assert (w_ref == kernel.w(r)) and np.all(dwdx_ref==kernel.dwdx(dx))
