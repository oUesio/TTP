'''
Individual class and solution representations
(Individual, random_tour, inheritance logic).
'''

class Individual:
    __slots__=("pi","z","obj")
    def __init__(self, pi, z, obj):
        self.pi=pi; self.z=z; self.obj=obj  # (time, -profit)
