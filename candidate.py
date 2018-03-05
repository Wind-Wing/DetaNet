import numpy as np

class Candidate:
    ''' A data struct to restore candidate struct info during evoluation algo. '''
    # evolution argus
        # range argus
    minFr = 4
    maxFr = 10
    minFc = 0
    maxFc = 3
    minM = 6
    maxM = 20
    minFl = 6 / 2       # due to filter number must be an even number for fire and demonsion layer
    maxFl = 20 / 2
        # mutation argus
    mutation_rate = 0.2

    # struct level 1
    feature_layer_num = 0  # feature abstract layers numbers
    feature_layer_array = [0 for i in range(feature_layer_num)] # 0 means conv layer, 1 means pool layer
    disable_mask = [0 for i in range(maxFr)]        # disable input is 1*1's Dimensionality_reduction_module
    fc_layer_num = 0       # full connection layers numbers
    # struct level 2
    module_num = 0         # modules in each layer
    # struct level 3
    filter_num = 0          # filters in each module, must be an even number
    

    # random generate candidates
    def __init__(self):
        self.feature_layer_num = self._get_random_int(self.minFr, self.maxFr)
        self.feature_layer_array = [np.random.randint(2) for i in range(self.feature_layer_num)] # 0 or 1 sequence
        self.fc_layer_num = self._get_random_int(self.minFc, self.maxFc)

        self.module_num = self._get_random_int(self.minM, self.maxM)

        self.filter_num = self._get_random_int(self.minFl, self.maxFl)

    
    def _get_random_int(self, min_value, max_value):
        '''
        # this random generator is based on that 
        # min_value is near 0
        # +-3*sigma value represent 99% possible
        _gap = max_value - min_value
        _value = np.random.normal(0,_gap/3.0,1)
        _value = int(abs(_value)) + min_value
        if _value > max_value:
            _value = max_value
        '''
        _value = np.random.randint(min_value, max_value + 1)
        return _value

    def mutation(self): # apply mutation to this candidate
        self.feature_layer_num = int((1 - self.mutation_rate) * self.feature_layer_num
                            + self.mutation_rate * self._get_random_int(self.minFr, self.maxFr))
        
        _dst = self.feature_layer_num - len(self.feature_layer_array)
        if(_dst < 0):
            for i in range(abs(_dst)):
                del self.feature_layer_array[np.random.randint(len(self.feature_layer_array))]
        elif(_dst > 0):
            self.feature_layer_array += [np.random.randint(2) for i in range(_dst)]

        for i in range(int(self.mutation_rate * len(self.feature_layer_array))):
            _index = np.random.randint(len(self.feature_layer_array))
            self.feature_layer_array[_index] = not self.feature_layer_array[_index] 

        self.fc_layer_num = int((1 - self.mutation_rate) * self.fc_layer_num 
                            + self.mutation_rate * self._get_random_int(self.minFc, self.maxFc))
        self.module_num = int((1 - self.mutation_rate) * self.module_num
                            + self.mutation_rate * self._get_random_int(self.minM, self.maxM))
        self.filter_num = int((1 - self.mutation_rate) * self.filter_num
                            + self.mutation_rate * self._get_random_int(self.minFl, self.maxFl))

    def crossover(self, parentA, parentB): # inheir genotype from parents
        _min = min(parentA.feature_layer_num ,parentB.feature_layer_num)
        cross_point = np.random.randint(_min)
        self.feature_layer_num = parentB.feature_layer_num
        self.feature_layer_array = parentA.feature_layer_array[:cross_point] + parentB.feature_layer_array[cross_point:]

        _min = min(parentA.fc_layer_num, parentB.fc_layer_num)
        _max = max(parentA.fc_layer_num, parentB.fc_layer_num)
        self.fc_layer_num = np.random.randint(_min, _max + 1)

        _min = min(parentA.module_num, parentB.module_num)
        _max = max(parentA.module_num, parentB.module_num)
        self.module_num = np.random.randint(_min, _max + 1)

        _min = min(parentA.filter_num, parentB.filter_num)
        _max = max(parentA.filter_num, parentB.filter_num)
        self.filter_num = np.random.randint(_min, _max + 1)

    def display_structure(self): # show network struct info represented by this candidate
        print("feature array, 0 - conv and 1 - pooling: ")
        print([self.feature_layer_array[i] for i in range(self.feature_layer_num) if self.disable_mask[i] != 1])
        print("full connection layer number: ")
        print(self.fc_layer_num)
        print("module number in each layer: ")
        print(self.module_num)
        print("filter number in each module: ")
        print(self.filter_num  * 2)
