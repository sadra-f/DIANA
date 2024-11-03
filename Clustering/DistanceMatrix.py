import numpy as np

class DistanceMatrix(object):
    """
    Distacnce Matrix class claculates the distance between each two values in the given dataset
    and stores the half matrix but provides the distance access for any two values in dataset     

    Args:
        object (_type_): _description_
    """
    def __init__(self, dataset, distance_function:callable):
        self._dataset = dataset
        self._distance_matrix = []
        self._distance_function = distance_function


    def build_DM(self):
        d_len = len(self._dataset) 
        for i in range(d_len):
            self._distance_matrix.append([])
            for j in range(i, d_len, 1):
                self._distance_matrix[i].append(self._distance_function(self._dataset[i], self._dataset[j]))

        
        return self
    

    def __getitem__(self, keys):
        try:
            # since half the dist matrix is kept we need to adjust the indices to the proper corresponding values
            #   0   1   2   3
            # 0 00 01  02  03
            # 1 10 11  12  13
            # 2 20 21  22  23
            # 3 30 31  32  33
            # TURNS INTO ==>
            #   0   1   2   3
            # 0 00 01  02  03
            # 1 11  12  13
            # 2 22  23
            # 3 33
            # which is why the lines below...
            row, col = keys
            if row > col :
                col, row = row, col
            col = col - row
            return self._distance_matrix[row][col]
        except TypeError:
            row = keys
            res = []
            for i in range(len(self._dataset)):
                res.append(self[row, i])
            return res