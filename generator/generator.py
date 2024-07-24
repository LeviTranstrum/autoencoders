import numpy
import scipy
import random
import os
import json

class Generator:
    @classmethod
    def random_rotation(cls):
        return scipy.spatial.transform.Rotation.from_euler(
           'xyz',
            [random.uniform(0.0, 2*numpy.pi),
              random.uniform(0.0, 2*numpy.pi),
                random.uniform(0.0, 2*numpy.pi)],
            degrees=False
        ).as_matrix().tolist()

    @classmethod
    def generate_data(cls):
        rotations = []
        for _ in range(1000000):
            rotations.append(Generator.random_rotation())
        data_path = './data/matrices.json'
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, mode="w") as data_file:
            json.dump(rotations, data_file, indent=2)