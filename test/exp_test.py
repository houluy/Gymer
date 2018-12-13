import unittest
import numpy as np
from solutions.cartpole import Experience


class TestExperience(unittest.TestCase):
    def setUp(self):
        self.observation = np.random.random((2, 2))
        self.action = np.random.random(1)
        self.rewards = np.random.random(1)
        self.y = [self.observation, self.action, self.rewards]
        self.e = Experience(self.observation, self.action, self.rewards)
        self.eiter = iter(self.e)

    def test_experience_iter(self):
        np.testing.assert_array_equal(next(self.eiter), self.observation)
        np.testing.assert_array_equal(next(self.eiter), self.action)
        np.testing.assert_array_equal(next(self.eiter), self.rewards)

    def test_experience_for(self):
        for x, y in zip(self.e, self.y):
            np.testing.assert_array_equal(x, y)


if __name__ == '__main__':
    unittest.main()
