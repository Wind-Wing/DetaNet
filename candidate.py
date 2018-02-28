import numpy as np 
class candidate:
    ''' A data struct to restore candidate struct info during evoluation algo. '''
    # struct level 1
    feature_layer_num = 0  # feature abstract layers numbers
    fc_layer_num = 0       # full connection layers numbers
    # struct level 2
    module_num = 0         # modules in each layer
    # struct level 3
    filter_num = 0          # filters in each module

    def __init__(self):pass # random generate candidates

    def mutation(self):pass # apply mutation to this candidate

    def crossover(self):pass# inheir genotype from parents
