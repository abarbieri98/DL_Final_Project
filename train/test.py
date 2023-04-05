import sys
sys.path.insert(1,"../")
import model as m
import torch
import torchvision
import numpy as np

if __name__ == "__main__":

    path_data = "../../COCO_sub"

    train_test = [8000, 2000]

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(256,256)),
        torchvision.transforms.CenterCrop(size=(224,224)),
        torchvision.transforms.ToTensor(),
        ])
    data = torchvision.datasets.ImageFolder(root=path_data, transform=data_transform) 
    _,test = torch.utils.data.random_split(data,train_test, generator=torch.Generator().manual_seed(42)) 
    np.random.seed(42)
    testloader = torch.utils.data.DataLoader(test, batch_size=8, pin_memory = True, num_workers = 4)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = m.UST_Net(level=1)
    model2 = m.UST_Net(level=2)
    model3 = m.UST_Net(level=3)
    model4 = m.UST_Net(level=4)
    model5 = m.UST_Net(level=5)
    model.load_state_dict(torch.load(r"../parameters/parameters_1"))
    model2.load_state_dict(torch.load(r"../parameters/parameters_2"))
    model3.load_state_dict(torch.load(r"../parameters/parameters_3"))
    model4.load_state_dict(torch.load(r"../parameters/parameters_4"))
    model5.load_state_dict(torch.load(r"../parameters/parameters_5"))
    model.to(device)
    model2.to(device)
    model3.to(device)
    model4.to(device)
    model5.to(device)

    loss = torch.nn.MSELoss(reduction="sum")
    test_loss = 0.
    
    for x,_ in iter(testloader):
        x=x.to(device)
        x_hat=model.forward(x)
        fx = model.encode(x)
        fx_hat = model.encode(x_hat)
        l = loss(x_hat, x) + loss(fx, fx_hat)
        test_loss += l.item()
    
    print("Test loss model 1:", test_loss/len(testloader.dataset))   
    test_loss = 0.
    
    for x,_ in iter(testloader):
        x=x.to(device)
        x_hat=model2.forward(x)
        fx = model2.encode(x)
        fx_hat = model2.encode(x_hat)
        l = loss(x_hat, x) + loss(fx, fx_hat)
        test_loss += l.item()
    print("Test loss model 2:", test_loss/len(testloader.dataset))   
    test_loss = 0.
    
    for x,_ in iter(testloader):
        x=x.to(device)
        x_hat=model3.forward(x)
        fx = model3.encode(x)
        fx_hat = model3.encode(x_hat)
        l = loss(x_hat, x) + loss(fx, fx_hat)
        test_loss += l.item()

    print("Test loss model 3:", test_loss/len(testloader.dataset))   
    test_loss = 0.
    
    for x,_ in iter(testloader):
        x=x.to(device)
        x_hat=model4.forward(x)
        fx = model4.encode(x)
        fx_hat = model4.encode(x_hat)
        l = loss(x_hat, x) + loss(fx, fx_hat)
        test_loss += l.item()
    
    print("Test loss model 4:", test_loss/len(testloader.dataset))   
    test_loss = 0.
    
    for x,_ in iter(testloader):
        x=x.to(device)
        x_hat=model5.forward(x)
        fx = model5.encode(x)
        fx_hat = model5.encode(x_hat)
        l = loss(x_hat, x) + loss(fx, fx_hat)
        test_loss += l.item()

    print("Test loss model 5:", test_loss/len(testloader.dataset))   

    
