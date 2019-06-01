from solutions.algorithms.ddpg import DeepDeterministicPolicyGradient
from solutions.games.cartpole import Cartpole
from solutions.algorithms.dqn import DQN

c = Cartpole()
# ddpg = DeepDeterministicPolicyGradient(
#     env=c,
# )
# ddpg.train(info=True)
# # c.run(c.random_policy, info=True)
# c.run(ddpg, info=True)
dqn = DQN(env=c)
dqn.train()
