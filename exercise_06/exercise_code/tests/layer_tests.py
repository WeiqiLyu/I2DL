import numpy as np

from .base_tests import UnitTest, CompositeTest, MethodTest, test_results_to_score
from exercise_code.networks.layer import *
from .gradient_check import eval_numerical_gradient_array


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class AffineForwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0
        num_inputs = 2
        input_shape = (4, 5, 6)
        output_dim = 3

        input_size = num_inputs * np.prod(input_shape)
        weight_size = output_dim * np.prod(input_shape)

        x = np.linspace(-0.1, 0.5,
                        num=input_size).reshape(num_inputs, *input_shape)
        w = np.linspace(-0.2, 0.3,
                        num=weight_size).reshape(np.prod(input_shape), output_dim)
        b = np.linspace(-0.3, 0.1, num=output_dim)

        self.value, _ = affine_forward(x, w, b)

        self.truth = np.array([[1.49834967, 1.70660132, 1.91485297],
                               [3.25553199, 3.5141327, 3.77273342]])

    def test(self):
        self.error = rel_error(self.truth, self.value)
        return (self.error < 1e-7)

    def define_failure_message(self):
        return "Affine forward incorrect. Expected: < 1e-7 Evaluated: " + str(self.error)


class AffineBackwardTestDx(UnitTest):
    def __init__(self):
        self.error = 0.0
        n = 10
        x = np.random.randn(n, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(n, 5)

        self.truth = eval_numerical_gradient_array(
            lambda x: affine_forward(x, w, b)[0], x, dout)

        _, cache = affine_forward(x, w, b)
        self.value, dw, db = affine_backward(dout, cache)

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-7

    def define_failure_message(self):
        return "Affine backward wrt x incorrect. Expected: < 1e-7 Evaluated: " + str(self.error)


class AffineBackwardTestDw(UnitTest):
    def __init__(self):
        self.error = 0.0
        n = 10
        x = np.random.randn(n, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(n, 5)

        self.truth = eval_numerical_gradient_array(
            lambda w: affine_forward(x, w, b)[0], w, dout) / n

        _, cache = affine_forward(x, w, b)
        dx, self.value, db = affine_backward(dout, cache)

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-7

    def define_failure_message(self):
        return "Affine backward wrt w incorrect. Expected: < 1e-7 Evaluated: " + str(self.error)


class AffineBackwardTestDb(UnitTest):
    def __init__(self):
        self.error = 0.0
        n = 10
        x = np.random.randn(n, 2, 3)
        w = np.random.randn(6, 5)
        b = np.random.randn(5)
        dout = np.random.randn(n, 5)

        self.truth = eval_numerical_gradient_array(
            lambda b: affine_forward(x, w, b)[0], b, dout) / n

        _, cache = affine_forward(x, w, b)
        dx, dw, self.value = affine_backward(dout, cache)

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-7

    def define_failure_message(self):
        return "Affine backward wrt b incorrect. Expected: < 1e-7 Evaluated: " + str(self.error)


class AffineLayerTest(CompositeTest):
    def define_tests(self):
        return [
            AffineForwardTest(),
            AffineBackwardTestDx(),
            AffineBackwardTestDw(),
            AffineBackwardTestDb()
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


class AffineTestWrapper:
    def __init__(self):
        self.affine_tests = AffineLayerTest()

    def __call__(self, *args, **kwargs):
        return "You secured a score of :" + \
            str(test_results_to_score(self.affine_tests()))


class SigmoidForwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0
        self.sigmoid = Sigmoid()
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

        self.value, _ = self.sigmoid.forward(x)
        self.truth = np.array([[0.37754067, 0.39913012, 0.42111892, 0.44342513],
                               [0.46596182, 0.48863832, 0.51136168, 0.53403818],
                               [0.55657487, 0.57888108, 0.60086988, 0.62245933]])

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-6

    def define_failure_message(self):
        return "Sigmoid forward incorrect. Expected: < 1e-6 Evaluated: " + str(self.error)


class SigmoidBackwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0
        self.sigmoid = Sigmoid()
        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)

        self.truth = eval_numerical_gradient_array(
            lambda x: self.sigmoid.forward(x)[0], x, dout)

        _, cache = self.sigmoid.forward(x)
        self.value = self.sigmoid.backward(dout, cache)

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-8

    def define_failure_message(self):
        return "Sigmoid backward incorrect. Expected: < 1e-8 Evaluated: " + str(self.error)


class SigmoidTest(CompositeTest):
    def define_tests(self):
        return [
            SigmoidForwardTest(),
            SigmoidBackwardTest()
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


class SigmoidTestWrapper:
    def __init__(self):
        self.sigmoid_tests = SigmoidTest()

    def __call__(self, *args, **kwargs):
        return "You secured a score of :" + \
            str(test_results_to_score(self.sigmoid_tests()))


class ReluForwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0
        self.relu = Relu()
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

        self.value, _ = self.relu.forward(x)
        self.truth = np.array([[0., 0., 0., 0., ],
                               [0., 0., 0.04545455, 0.13636364],
                               [0.22727273, 0.31818182, 0.40909091, 0.5]])

    def test(self):
        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-6

    def define_failure_message(self):
        return "Relu forward incorrect. Expected: < 1e-6 Evaluated: " + str(self.error)


class ReluBackwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0
        self.relu = Relu()
        x = np.random.randn(10, 10)
        d = np.ones_like(x)
        d[x <= 0] = 0
        dout = np.random.randn(*x.shape)

        self.truth = dout * d

        _, cache = self.relu.forward(x)
        self.value = self.relu.backward(dout, cache)

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-8

    def define_failure_message(self):
        return "Relu backward incorrect. Expected: < 1e-8 Evaluated: " + str(self.error)


class ReluTest(CompositeTest):
    def define_tests(self):
        return [
            ReluForwardTest(),
            ReluBackwardTest()
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


class ReluTestWrapper:
    def __init__(self):
        self.relu_tests = ReluTest()

    def __call__(self, *args, **kwargs):
        return "You secured a score of :" + \
            str(test_results_to_score(self.relu_tests()))


class LeakyReluForwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0
        self.lrelu = LeakyRelu()
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)
        self.value, _ = self.lrelu.forward(x)
        self.truth = np.array([[-0.5 * self.lrelu.slope, -0.40909091 * self.lrelu.slope,
                                -0.31818182 * self.lrelu.slope, -0.22727273 * self.lrelu.slope],
                               [-0.13636364 * self.lrelu.slope, -0.04545455 * self.lrelu.slope, 0.04545455,
                                0.13636364],
                               [0.22727273, 0.31818182, 0.40909091, 0.5]])

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-6

    def define_failure_message(self):
        return "LeakyRelu forward incorrect. Expected: < 1e-6 Evaluated: " + str(self.error)


class LeakyReluBackwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0
        self.lrelu = LeakyRelu()

        x = np.random.randn(10, 10)
        d = np.ones_like(x)
        d[x <= 0] = self.lrelu.slope
        dout = np.random.randn(*x.shape)
        self.truth = dout * d

        _, cache = self.lrelu.forward(x)
        self.value = self.lrelu.backward(dout, cache)

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-8

    def define_failure_message(self):
        return "LeakyRelu backward incorrect. Expected: < 1e-8 Evaluated: " + str(self.error)


class LeakyReluTest(CompositeTest):
    def define_tests(self):
        return [
            LeakyReluForwardTest(),
            LeakyReluBackwardTest()
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


class LeakyReluTestWrapper:
    def __init__(self):
        self.lrelu_tests = LeakyReluTest()

    def __call__(self, *args, **kwargs):
        return "You secured a score of :" + \
            str(test_results_to_score(self.lrelu_tests()))


class TanhForwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0
        self.tanh = Tanh()
        x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

        self.value, _ = self.tanh.forward(x)
        self.truth = np.array([[-0.46211716, -0.38770051, -0.30786199, -0.22343882],
                               [-0.13552465, -0.04542327, 0.04542327, 0.13552465],
                               [0.22343882, 0.30786199, 0.38770051, 0.46211716]])

    def test(self):

        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-6

    def define_failure_message(self):
        return "Tanh forward incorrect. Expected: < 1e-6 Evaluated: " + str(self.error)


class TanhBackwardTest(UnitTest):
    def __init__(self):
        self.error = 0.0
        self.tanh = Tanh()

        x = np.random.randn(10, 10)
        dout = np.random.randn(*x.shape)

        self.truth = eval_numerical_gradient_array(
            lambda x: self.tanh.forward(x)[0], x, dout)

        _, cache = self.tanh.forward(x)
        self.value = self.tanh.backward(dout, cache)

    def test(self):
        self.error = rel_error(self.truth, self.value)

        return self.error < 1e-8

    def define_failure_message(self):
        return "Tanh backward incorrect. Expected: < 1e-8 Evaluated: " + str(self.error)


class TanhTest(CompositeTest):
    def define_tests(self):
        return [
            TanhForwardTest(),
            TanhBackwardTest()
        ]

    def define_success_message(self):
        return "Congratulations you have passed all the unit tests!!!"

    def define_failure_message(self):
        return "Test cases are still failing!"


class TanhTestWrapper:
    def __init__(self):
        self.tanh_tests = TanhTest()

    def __call__(self, *args, **kwargs):
        return "You secured a score of :" + \
            str(test_results_to_score(self.tanh_tests()))
