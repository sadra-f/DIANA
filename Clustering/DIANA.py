from .DistanceMatrix import DistanceMatrix as DM

class DIANA:
    def __init__(self, steps:str|int, distance_func:callable):
        self.steps = steps
        self.distance_func = distance_func
        self._dists = None


    def fit(self, dataset):
        self.dataset = dataset
        self._dists = DM(self.dataset, self.distance_func).build_DM()
