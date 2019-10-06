import numpy as np

def f(a):
    def g(c):
        print c \
                    +a
    return g
h = f(2)
h(233)
