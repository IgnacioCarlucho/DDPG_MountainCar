import random
import numpy as np 

class OU(object):

    def function(self, x, mu, theta, sigma, value):
        return theta * (mu - x) + value*sigma * np.random.randn(1)