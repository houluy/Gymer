from solutions.algorithms.ddpg import DeepDeterministicPolicyGradient
from solutions.games.cartpole import Cartpole

c = Cartpole()
ddpg = DeepDeterministicPolicyGradient(
    env=c,
)
ddpg.train()
c.run(ddpg)
