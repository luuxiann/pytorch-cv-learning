import torch

A1 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
B1 = A1.T
C1 = B1.T
print("1")
print(A1 == C1)

A2 = torch.arange(9, dtype=torch.float32).reshape(3, 3)
B2 = A2.T
C2 = A2.T
D2 = B2.T
E2 = (A2+B2).T
print("2")
print(E2)
print(C2+D2)
print(E2 ==(C2+D2))

A3 = torch.arange(9, dtype=torch.float32).reshape(3, 3)
B3 = A3 + A3.T
print("3")
print(B3)

A4 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print("4")
print(len(A4))

A5 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
B5 = torch.arange(12, dtype=torch.float32).reshape(4, 3)
print("5")
print(len(A5))
print(len(B5))

A6 = torch.arange(9, dtype=torch.float32).reshape(3, 3)
print("6")
B6 = A6/A6.sum(axis=1)
print(B6)

A7 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print("7")
print(A7.sum(axis=0))
print(A7.sum(axis=1))
print(A7.sum(axis=2))

A8 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
B8 = torch.arange(12, dtype=torch.float32).reshape(4, 3)
C8 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print("8")
print(A8)
print(B8)
print(C8)