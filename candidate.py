import numpy as np

class Candidate:
    ''' A data struct to restore candidate struct info during evoluation algo. '''
    # struct level 1
    feature_layer_num = 0  # feature abstract layers numbers
    fc_layer_num = 0       # full connection layers numbers
    feature_layer_array = [0 for i in range(feature_layer_num)] # 0 means conv layer, 1 means pool layer
    # struct level 2
    module_num = 0         # modules in each layer
    # struct level 3
    filter_num = 0          # filters in each module

    # random generate candidates
    def __init__(self):
        self.feature_layer_num = 2
        self.fc_layer_num = 1
        self.feature_layer_array = [0,1]
        self.module_num = 20
        self.filter_num = 20

    def mutation(self):pass # apply mutation to this candidate

    def crossover(self):pass# inheir genotype from parents
