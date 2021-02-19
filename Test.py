import random
import unittest
import Moran_Process as mp
import Graphs

class test_moran_process(unittest.TestCase):

    def test_mutation_of_random_node(self):
        G = Graphs.createCompleteGraph()
        random.seed(1)
        mp.mutate_a_random_node(G)
        self.assertEqual(G.nodes[1]['type'].id_n,'mutant')

if __name__ == '__main__':
    unittest.main()