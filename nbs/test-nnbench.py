import unittest
from unittest.mock import patch, MagicMock, call
from pprint import pprint, pformat

import numpy as np
from numpy import array
import sympy

from nnbench import NNBench
import nn

def arangep(n, starting_index=0):
    sympy.sieve.extend_to_no(starting_index + n)
    return np.array(sympy.sieve._list[starting_index:starting_index + n])


class TestNNBench(unittest.TestCase):
    def create_patch(self, name):
        patcher = patch(name)
        thing = patcher.start()
        self.addCleanup(patcher.stop)
        return thing

    def setUp(self):
        self.create_patch('nn.Network')


    def test_bench_checkpoint(self):
        net = nn.Network()
        bench = NNBench(net)

        # Checkpointing a net saves the internal state vector
        sv = 'some state vector'
        net.state_vector.return_value = sv
        bench.checkpoint_net()
        net.state_vector.assert_called_once()

        # Rolling back a net sets the net's state vector from the saved value
        bench.rollback_net()
        net.set_state_from_vector.assert_called_once_with(sv)


    def test_bench_network_input_width_detection(self):
        # Mock up a network with a determined input width
        net = nn.Network()
        mock_layer = MagicMock()
        mock_layer.M = np.zeros(2*3).reshape(3,2)
        net.layers = [mock_layer]

        # A bench of a net discovers the input width of the net
        bench = NNBench(net)
        self.assertEqual(bench.net.layers[0].M.shape, (3,2)) # verify we mocked it as intended
        self.assertEqual(NNBench.net_input_width(bench.net), 2)


    def test_network_learn_input_form(self):
        # Make a constant training batch for a two-in, three-out net,
        # containing two examples
        training_batch = (np.arange(2*3).reshape(-1,2), arangep(3*3).reshape(-1,3))

        # The training batch matches the form expected by Network.learn
        facts = [training_batch, training_batch] # Facts can have multiple batches

        # Duplicating the looper from Network.learn
        for x, expected in facts:
            self.assertEqual(x.shape[0], expected.shape[0]) # Each input has an output
            self.assertEqual(x.shape[1], 2) # Inputs have width 2
            self.assertEqual(expected.shape[1], 3) # Outputs have width 3

    def test_bench_learn_training_batch(self):
        # Mock up a network of input width 2
        net = nn.Network()
        mock_layer = MagicMock()
        mock_layer.M = np.zeros(2*3).reshape(3,2)
        net.layers = [mock_layer]
        bench = NNBench(net)

        # Make a constant training batch for a two-in, three-out net,
        # containing two examples
        training_batch = (np.arange(2*3).reshape(-1,2), arangep(3*3).reshape(-1,3))

        # The training batch matches the form expected by Network.learn
        facts = [training_batch, training_batch] # Facts can have multiple batches
        for x, expected in facts:
            self.assertEqual(x.shape[0], expected.shape[0]) # Each input has an output
            self.assertEqual(x.shape[1], 2) # Inputs have width 2
            self.assertEqual(expected.shape[1], 3) # Outputs have width 3

        bench.training_batch = lambda n: training_batch
        #print(bench.training_batch)
        bench.learn(batches=2)
        expected = [call([(array([[0, 1],
                                  [2, 3],
                                  [4, 5]]),
                          array([[ 2,  3,  5],
                                 [ 7, 11, 13],
                                 [17, 19, 23]]))]),
                    call([(array([[0, 1],
                                  [2, 3],
                                  [4, 5]]),
                          array([[ 2,  3,  5],
                                 [ 7, 11, 13],
                                 [17, 19, 23]]))])]
        self.assertEqual(pformat(expected), pformat(net.learn.mock_calls))


    def test_training_data_batching_from_literal(self):
        net = nn.Network()

        # Setup for input width of 2
        mock_layer = MagicMock()
        mock_layer.M = np.zeros(2*3).reshape(3,2)
        net.layers = [mock_layer]
        bench = NNBench(net)

        # A bench can accept literal Facts
        training_data = [([1,2],[5]), ([4,3], [10])] # Two truths
        bench.accept_source_of_truth(training_data)
        bench.training_batch = bench.training_batch_from_gen

        # It feeds the net batches of the requested size, produced from Facts, by repetition
        bench.learn(batches=2, batch_size=3)
        #print(net.mock_calls)
        expected = [call([(array([[1, 2], [4, 3], [1, 2]]), array([[ 5], [10], [ 5]]))]),
                    call([(array([[4, 3], [1, 2], [4, 3]]), array([[10], [ 5], [10]]))])]
        self.assertEqual(pformat(expected), pformat(net.learn.mock_calls))


    def test_training_data_batching_from_gen(self):
        net = nn.Network()

        # Setup for input width of 2
        mock_layer = MagicMock()
        mock_layer.M = np.zeros(2*3).reshape(3,2)
        net.layers = [mock_layer]
        bench = NNBench(net)

        # A bench can accept generated facts
        bench.accept_source_of_truth(((v, np.array([v.dot(np.array([2, 3]))])) for v in (arangep(2,2*i) for i in range(5))))

        # It feeds the net batches of the requested size, produced from Facts, by repetition
        net.reset_mock()
        bench.learn(batches=2, batch_size=3)
        #print(net.mock_calls)
        expected = [call([(array([[2, 3], [5, 7], [11, 13]]), array([[ 13], [31], [61]]))]),
                    call([(array([[17, 19], [23, 29], [2, 3]]), array([[91], [133], [13]]))])]
        self.assertEqual(pformat(expected), pformat(net.learn.mock_calls))


    @unittest.skip("Test is WIP")
    def test_bench_learn_training_data_gen_fixed(self):
        net = nn.Network()

        # Setup for input width of 2
        mock_layer = MagicMock()
        mock_layer.M = np.zeros(2*3).reshape(3,2)
        net.layers = [mock_layer]
        bench = NNBench(net)


        #bench.ideal = lambda v: v @ np.array([0,1,1,0]).reshape(2,2) #UNUSED in this test
        bench.training_data = [([1,2],[5]),
                               ([4,3], [10])]
        bench.training_data_gen = bench.training_data_gen_fixed
        #pprint(list(bench.training_data_gen(3)))
        bench.learn(batches=3, batch_size=2)
        #print(net.mock_calls)
        expected = [call([(array([[1, 2], [4, 3]]), array([[ 5], [10]]))]),
                    call([(array([[1, 2], [4, 3]]), array([[ 5], [10]]))]),
                    call([(array([[1, 2], [4, 3]]), array([[ 5], [10]]))])]
        #self.assertEqual(net.learn.mock_calls, expected)
        #FIXME how? above does not work in Python 3.7.9.
        # Maybe when the gpu-jupyter docker container gets updated
        # The below is a classic and embarassing hack that gets us around this deficiency
        self.assertEqual(pformat(expected), pformat(net.learn.mock_calls))

        # Now do it with a batch size different from the fixed training data length
        # It should cycle through the training data
        net.reset_mock()
        bench.learn(batches=2, batch_size=3)
        print(net.mock_calls)
        expected = [call([(array([[1, 2], [4, 3], [1, 2]]), array([[ 5], [10], [ 5]]))]),
                    call([(array([[4, 3], [1, 2], [4, 3]]), array([[10], [ 5], [10]]))])]
        self.assertEqual(pformat(expected), pformat(net.learn.mock_calls))


    @unittest.skip("Test is defective")
    def test_bench_learn_gen_randn(self):
        net = nn.Network()
        mock_layer = MagicMock()
        mock_layer.M = np.zeros(2*3).reshape(3,2)
        net.layers = [mock_layer]
        bench = NNBench(net)

        def adc(input):
            m = max(0, min(7, int(8*input)))
            return np.array([(m>>2)&1, (m>>1)&1, m&1]) - 0.5

        vadc = lambda v: np.array([adc(p) for p in v])

        def training_data_gen_randn(n):
            """Generate n instances of labelled training data"""
            width = bench.input_width
            v = np.random.random_sample((n,1)) * 1.2 - 0.1
            yield v, vadc(v)

        bench.training_data_gen = training_data_gen_randn
        loss = bench.learn(batches=2)
        expected = 'fur'
        self.assertEqual(pformat(expected), pformat(net.learn.mock_calls))



class MyTest(unittest.TestCase):
    def setUp(self):
        patcher = patch('nn.Network')
        self.MockClass = patcher.start()
        self.addCleanup(patcher.stop)

    def test_something(self):
        assert nn.Network is self.MockClass

    def test_other(self):
        assert nn.Network is self.MockClass

class DemoTest(unittest.TestCase):
    def create_patch(self, name):
        patcher = patch(name)
        thing = patcher.start()
        self.addCleanup(patcher.stop)
        return thing

    def test_foo(self):
        mock_Network = self.create_patch('nn.Network')
        mock_AffineLayer = self.create_patch('nn.AffineLayer')
        mock_MapLayer = self.create_patch('nn.MapLayer')

        assert nn.Network is mock_Network
        assert nn.AffineLayer is mock_AffineLayer
        assert nn.MapLayer is mock_MapLayer


if __name__ == '__main__':
    unittest.main()
