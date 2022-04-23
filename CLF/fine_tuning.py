import torch
import torchvision


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.fc = torch.nn.Linear(in_features=1 * 6 * 30 * 30, out_features=2)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = x.view(-1, 1*6*30*30)
        x = self.fc(x)
        x = torch.nn.functional.softmax(x)
        return x

if __name__ == '__main__':
    print(torch.__version__)
    # net = torchvision.models.resnet18(pretrained=True)
    net = Net()
    net.eval()
    print("Net Prediction:", net(torch.rand((1, 3, 32, 32))))
    # for params in net.parameters():
    #     params.requires_grad = False

    head = torch.nn.Linear(in_features=net.fc.in_features, out_features=2)
    net.fc = head
    # print(net)
    print(net(torch.rand((1, 3, 32, 32))))
    x_ones = torch.ones((64, 3, 32, 32))
    y_ones = torch.ones((64, 2))
    y_ones[:, 0] = 0
    # print(y_ones)
    x_zeros = torch.zeros((64, 3, 32, 32))
    y_zeros = torch.zeros((64, 2))
    y_zeros[:, 0] = 1
    # print(y_zeros)

    x = torch.cat([x_ones, x_zeros])
    y = torch.cat([y_ones, y_zeros])

    epochs =5000
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    for e in range(0, epochs):
        prds = net(x)
        loss = loss_fn(y, prds)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if e % 100 ==0:
            for param in net.parameters():
                print("Requires_Grad:", param.requires_grad)
            print("Loss:", loss, " L:", y[-1], " P:", prds[-1])
        # break

    # print(y_ones[0], " : ", y_zeros[0])




