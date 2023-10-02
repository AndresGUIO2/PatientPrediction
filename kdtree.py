from pydantic import BaseModel
from typing import List, Tuple
from scipy.spatial import KDTree
import numpy as np

class KnnQuery(BaseModel):
    new_point: list[float]
    k: int

class BallQuery(BaseModel):
    new_point: list[float]
    radius: float

class KDTreeDS:
    def __init__(self, points):
        self.tree = KDTree(points)

    def knn_query(self, point: List[float], k: int) -> List[Tuple[float, List[float], int]]:
        distances, indexes = self.tree.query(np.array([point]), k=k)
        indexes = indexes.flatten()
        distances = distances.flatten()
        print("Indices:", indexes)  #debugging
        print("Distances:", distances)  #debugging

        return indexes, distances

    def ball_query(self, point: List[float], radius: float) -> List[ int]:
        indexes_list = self.tree.query_ball_point(np.array([point]), radius, p= 2)

        inner_list = []
        if indexes_list:
            inner_list = indexes_list[0][0] if indexes_list[0] else []

        return inner_list
        
