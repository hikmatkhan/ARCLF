import torch
import torchvision


if __name__ == '__main__':
    print(torch.__version__)
    resnet = torchvision.models.resnet18(pretrained=True)
    print(resnet)

    epochs = 10000
    rand_sample = torch.ones((1, 3, 64, 64))#rand((1, 3, 64, 64))
    label = torch.zeros((1, 1000))
    label[0, 0] = 1
    optim = torch.optim.SGD(resnet.parameters(), lr=0.00001)
    for e in range(0, epochs):
        prediction = resnet(rand_sample)
        print(torch.sum(prediction))
        break
        loss = (label - prediction).sum() # Error tensor
        loss.backward()
        optim.step()
        optim.zero_grad()
        if e % 500:
            print("L:", label[0, 0], " P:", prediction[0, 0], " Loss:", loss.item())



