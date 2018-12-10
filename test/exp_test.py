import unittest
import numpy as np
from solutions.cartpole import Experience, ExperiencePool


class TestExperience(unittest.TestCase):
    def setUp(self):
        self.observation = np.random.random((2, 2))
        self.action = np.random.random(1)
        self.rewards = np.random.random(1)
        self.n = np.random.random((2, 2))
        self.y = [self.observation, self.action, self.rewards, self.n]
        self.e = Experience(self.observation, self.action, self.rewards, self.n)
        self.eiter = iter(self.e)

    def test_experience_iter(self):
        np.testing.assert_array_equal(next(self.eiter), self.observation)
        np.testing.assert_array_equal(next(self.eiter), self.action)
        np.testing.assert_array_equal(next(self.eiter), self.rewards)
        np.testing.assert_array_equal(next(self.eiter), self.n)

    def test_experience_for(self):
        for x, y in zip(self.e, self.y):
            np.testing.assert_array_equal(x, y)


class TestExpPool(unittest.TestCase):
    def setUp(self):
        self.size = 5
        self.pool = ExperiencePool(self.size)
        self.observation = np.random.random((self.size, 2, 2))
        self.action = np.random.random((self.size, 1))
        self.rewards = np.random.random((self.size, 1))
        self.n = np.random.random((self.size, 2, 2))

    def test_pool_set(self):
        try:
            for n in range(self.size):
                self.pool[n] = [self.observation[n], self.action[n], self.rewards[n], self.n[n]]
        except (TypeError, ValueError) as e:
            self.fail(e)
        finally:
            self.pool.clear()

    def test_pool_append(self):
        try:
            for n in range(self.size):
                self.pool.append([self.observation[n], self.action[n], self.rewards[n], self.n[n]])
        except (TypeError, ValueError) as e:
            self.fail(e)
        finally:
            self.pool.clear()

    def test_pool_get(self):
        for n in range(self.size):
            self.pool[n] = [self.observation[n], self.action[n], self.rewards[n], self.n[n]]
        for n in range(self.size):
            exp = self.pool[n]
            for ind, e in enumerate(exp):
                np.testing.assert_array_equal(e, [self.observation[n], self.action[n], self.rewards[n], self.n[n]][ind])

    def test_pool_get_by_slice(self):
        for n in range(self.size):
            self.pool[n] = [self.observation[n], self.action[n], self.rewards[n], self.n[n]]

        observations, actions, rewards, n = self.pool[:]
        self.observation.tolist()
        observations.tolist()
        print(self.observation)
        print(observations)
        self.assertCountEqual(self.observation, observations)
        # np.testing.assert_array_equal(np.sort(self.action), np.sort(actions))
        # np.testing.assert_array_equal(np.sort(self.rewards), np.sort(rewards))
        # np.testing.assert_array_equal(np.sort(self.n), np.sort(n))


if __name__ == '__main__':
    unittest.main()
