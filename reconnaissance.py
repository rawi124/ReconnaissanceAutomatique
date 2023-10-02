import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot
from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs
from pylab import *
from numpy import *

b = open("0.raw", "r")
b.decode('UTF-8')
print(b.readlines())
