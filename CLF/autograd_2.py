import torch


if __name__ == '__main__':
    print(torch.__version__)
    a = torch.tensor([[1, 2]], dtype=torch.float, requires_grad=True)
    b = torch.tensor(([[3, 4]]), dtype=torch.float, requires_grad=True)
    Q = 3*a**3-b**2
    print("A:", a, " B:", b)
    print("A_Grad:", a.grad, " B_Grad:", b.grad)
    Q.sum().backward()
    print("Q:", Q)
    print("A_Grad:", a.grad, " B_Grad:", b.grad)




