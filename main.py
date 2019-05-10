from solutions.algorithms.ddpg import DeepDeterministicPolicyGradient
from solutions.games.cartpole import Cartpole

c = Cartpole()
ddpg = DeepDeterministicPolicyGradient(
    env=c,
)
ddpg.train(info=True)
# c.run(c.random_policy, info=True)
c.run(ddpg, info=True)
