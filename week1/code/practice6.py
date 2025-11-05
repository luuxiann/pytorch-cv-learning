import torch
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([10]) / 10
counts = multinomial.Multinomial(10, fair_probs).sample((500,))   
cum_counts = counts.cumsum(dim=0)   
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)   

d2l.set_figsize((10, 10))
for i in range(10):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.1, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()