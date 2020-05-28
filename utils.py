import numpy as np
import random 

def gen_neg_samples_with_npratio(neg_array, ratio):

    if ratio > len(neg_array):
        return random.sample(list(neg_array)*(ratio//len(neg_array)+1), ratio)
    else:
        return random.sample(list(neg_array), ratio)
