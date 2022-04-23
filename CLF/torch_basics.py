import torch
import numpy as np
if __name__ == '__main__':

    # Tensor vs Numpy
    data = [[1,2], [3,4]]
    t_data = torch.tensor(data)
    print(t_data)
    n_data = np.array(data)
    print(n_data)

    # Numpy to tensor
    print(torch.from_numpy(n_data))

    # From another tensor
    print(torch.ones_like(t_data, dtype=torch.float))

    # From constant or random to tensor
    rand_tensor = torch.rand((2,2))
    print(rand_tensor)
    const_tensor = torch.ones((2,2))
    print(const_tensor)
    zero_tensor = torch.zeros((2,2))
    print(zero_tensor)

    # Tensor Attributes
    print("Shape:", rand_tensor.shape, " dtype:", rand_tensor.dtype, " device:", rand_tensor.device)

    # Check cuda availability
    if torch.cuda.is_available():
        print("Yes, cuda acceleration is available")
    else:
        print("No, cuda is not available")

    # Assigning tensor values
    ones_tensor = torch.ones((4,4))
    ones_tensor[:, 1] = -1
    print(ones_tensor)

    # Joining two tensors
    ct_1 = torch.ones((3,3))
    ct_2 = torch.zeros((3,3))
    ct_3 = torch.zeros(((3,3)))
    print("T:", ct_1)
    print("T2: Dim0", torch.cat([ct_1, ct_2], dim=0), torch.cat([ct_1, ct_2], dim=0).shape)
    print("T2: Dim0", torch.cat([ct_1, ct_2], dim=1), torch.cat([ct_1, ct_2], dim=1).shape)
    print("T1T2T3:", torch.cat([ct_1, ct_2,ct_3]))

    # Multiplication
    tensor = torch.tensor([[1,9,4],[5,1,1]])
    print(torch.mul(tensor, tensor)) # Elementwise multiplication
    print(torch.matmul(tensor, tensor.T))
    print(torch.matmul(tensor.T, tensor))

    # In-place operations
    tensor.add_(100)
    print("In-Place:", tensor)
    # Bridging the numpy
    np_data = tensor.numpy()
    print("NP:", np_data)



