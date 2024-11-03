from .DistanceMatrix import DistanceMatrix as DM
import numpy as np
class DIANA:
    """
    Divisive Analysis Clustering a Hierarchical Top-Down Clustering Algorithm

    """
    def __init__(self, steps:int|None, distance_func:callable):
        self.step_limit = steps
        self.distance_func = distance_func
        self._dists = None
        # self._clusters = None
        self.labels = None
        self._cluster_history = None
        self._separated = None


    def _cluster(self):
        selected_cluster_index = 0
        new_cluster_index = -1
        step_count = 0
        while True:
            print(step_count)
            if self.step_limit is not None:
                if step_count >= self.step_limit:
                    break
            #iter 3 or 1 ==> select which cluster needs to be splitted
            diameters = []
            for i in self._separated:
                diameters.append(self._cluster_diameter(i))
            if max(diameters) <= 0: break
            selected_cluster_index = diameters.index(max(diameters))

            # iter 1 or 2 ==> check which element in selected cluster should build a new cluster
            intra_dissim_res = self._max_intra_dissimilarity(selected_cluster_index)
            # since the len is one larger than the last index which will be the last index if we append a new item
            new_cluster_index = len(self._separated)
            self._separated.append([self._separated[selected_cluster_index][intra_dissim_res['Index']]])
            del self._separated[selected_cluster_index][intra_dissim_res['Index']]

            # iter 2 or 3 ==> find and move elements from selected cluster to new cluster
            dissim_res = self._max_inter_dissimilarity(selected_cluster_index, new_cluster_index)
            while dissim_res is not None:
                self._separated[new_cluster_index].append(self._separated[selected_cluster_index][dissim_res['Index']])
                del self._separated[selected_cluster_index][dissim_res['Index']]
                dissim_res = self._max_inter_dissimilarity(selected_cluster_index, new_cluster_index)
            self._add_history()
            
            step_count += 1
        
        return 
        
    def cluster(self, dataset):
        self.dataset = dataset
        self._d_len = len(self.dataset)
        self._dists = DM(self.dataset, self.distance_func).build_DM()
        self._separated = [[i for i in range(self._d_len)]]
        self._cluster_history = []
        self._add_history()
        self._cluster()
        self._build_labels()
        return self

    def _max_intra_dissimilarity(self, cluster_index):
        """finds the element which is most dissimilar in its cluster

        Args:
            cluster_index (int): the index of the cluster in question

        Returns:
            dict: index and avg of the element most dissimilar
        """
        max_avg = -1
        max_indx = -1
        for i, val1 in enumerate(self._separated[cluster_index]):
            local_sum = 0
            local_avg = 0
            for j, val2 in enumerate(self._separated[cluster_index]):
                local_sum += self._dists[val1][val2]
            local_avg = local_sum / (len(self._separated[cluster_index]) - 1)
            if local_avg > max_avg:
                max_avg = local_avg
                max_indx = i
        return {'Index':max_indx, 'Average':max_avg}

    def _max_inter_dissimilarity(self, cluster1_index, cluster2_index):
        """finds which element from cluster 1 can be in cluster 2

        Args:
            cluster1_index (int): index of cluster 1
            cluster2_index (int): index of cluster 2

        Returns:
            dict: the index and dissimilarity value of the element or None if no dissim value is above 0
        """
        max_dissim_val = -1
        max_dissim_index = -1
        for i, val1 in enumerate(self._separated[cluster1_index]):
            local_intra_sum = 0
            local_intra_avg = 0

            local_inter_sum = 0
            local_inter_avg = 0

            local_dissim_val = -1
            for j, val2 in enumerate(self._separated[cluster1_index]):
                local_intra_sum += self._dists[val1][val2]
            local_intra_avg = local_intra_sum / (len(self._separated[cluster1_index]) - 1)
            for j, val2 in enumerate(self._separated[cluster2_index]):
                local_inter_sum += self._dists[val1][val2]
            local_inter_avg = local_inter_sum / len(self._separated[cluster2_index])
            local_dissim_val =  local_intra_avg - local_inter_avg
            if local_dissim_val > max_dissim_val:
                max_dissim_val = local_dissim_val
                max_dissim_index = i
        if max_dissim_val > 0:
            return {'Index':max_dissim_index, 'Value':max_dissim_val}
        return None

    def _add_history(self):
        self._cluster_history.append(list(self._separated))

    def _cluster_diameter(self, cluster):
        """finds how similar values in the cluster are to each other

        Args:
            cluster (list): cluster(list) with indices of dataset values as cluster values.

        Returns:
            float: the diameter (similarity) of elements in cluster
        """
        clus_sum = 0
        for i, val1 in enumerate(cluster):
            for j, val2 in enumerate(cluster[i:]):
                clus_sum += self._dists[val1][val2]
        
        return clus_sum / len(cluster)
    
    def _build_labels(self):
        self.labels = -1 * np.ones((self._d_len,), int)
        for i, val1 in enumerate(self._separated):
            for j, val2 in enumerate(val1):
                self.labels[val2] = i