import torch

t1 = torch.Tensor([5,2,6,7])
t2 = torch.Tensor([9,3,6,1])
l1 = [5,2,6,7]
l2 = [9,3,6,1]

print(t1+t2)
print(l1+l2)

conc = torch.stack((t1,t2), dim=0)
print(conc)

t3 = torch.rand((4,2))
print(t3.shape)
print(conc.shape)

# Matrix Multiplication
mult = torch.mm(conc, t3)
print(mult)

# Slicing
print(mult[:,0])
