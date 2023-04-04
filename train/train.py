import sys
sys.path.insert(1,"../")
import model as m
import torch
import torchvision
import numpy as np
import argparse
from time import time

parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int,default=32, help='batch size')
parser.add_argument("--epoch", type=int,default=15, help='number of epochs for the training')
parser.add_argument("--lr", type=float,default=1e-4, help='learning rate')
parser.add_argument("--modelName", type = str, default = "saved_parameters", help= "File name for the trained model (usually [class of the model]_[n epochs])")
parser.add_argument("--lam", type = float, default = 1, help= "Lambda parameter for loss function")
parser.add_argument("--subset", type = int, default = 0, help= "Subset dataset for training (0 for full dataset)")
parser.add_argument("--level", type=int, default=5, help="Select the the depth of the model.")
parser.add_argument("--partial", action='store_true', default=False, help="Activate partial training, saving the state of the model and optimizer at the end.")
parser.add_argument("--loadCheckpoint", type= str, default="", help="Checkpoint name to load")

opt = parser.parse_args()
if __name__ == "__main__":
    ######### Import data and transforms #########

    # Here we define the import pipeline and split between train and test set.
    # We chose as train/test ratio 80/20

    path_data = "../../COCO_sub/"
    train_test = [8000, 2000]

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(256,256)),
        torchvision.transforms.CenterCrop(size=(224,224)),
        torchvision.transforms.ToTensor(),
        ])
    data = torchvision.datasets.ImageFolder(root=path_data, transform=data_transform) 

    train,_ = torch.utils.data.random_split(data,train_test, generator=torch.Generator().manual_seed(42)) 

    if opt.subset > 0:
        train_subset = int(np.ceil(opt.subset * .8))
        np.random.seed(42)
        idx_train = np.random.choice(np.arange(0,stop = train_test[0]),train_subset)
        subtrain = torch.utils.data.Subset(train,idx_train)
        trainloader = torch.utils.data.DataLoader(subtrain, batch_size=opt.batchSize, pin_memory = True, num_workers = 4)
    else:
        trainloader = torch.utils.data.DataLoader(train, batch_size=opt.batchSize,pin_memory = True, num_workers = 4 )
    ######### Training #########

    # Here we define the object needed for the training and the training itself.
    # The code was developed in order to work in both CPU and GPU, even though GPU is highly recommended.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = m.UST_Net(level=opt.level)
    # Fix the VGG19 parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scaler = torch.cuda.amp.GradScaler() # needed to implement autocasting
    loss = torch.nn.MSELoss(reduction="sum")
    epochs = opt.epoch
    lam = opt.lam
    prev_epochs = 0

    if opt.loadCheckpoint:
        prev_check = torch.load(opt.loadCheckpoint)
        model.load_state_dict(prev_check['model_state_dict'])
        optimizer.load_state_dict(prev_check['optimizer_state_dict'])
        prev_epochs = prev_check['epoch']

    st = time()
    for epoch in np.arange(epochs):
        model.train()
        train_loss = 0
        for x,_ in iter(trainloader):
            x = x.to(device)
            x_hat=model.forward(x)
            fx = model.encode(x)
            fx_hat = model.encode(x_hat)
            with torch.autocast(device_type = "cuda", dtype = torch.float16):
                # loss is computed using float16 type instead of float32, 
                # improving performance and runtime of  the model at the cost
                # of an approximation of the gradient.
                l=loss(x_hat,x) + lam*loss(fx,fx_hat)
                train_loss+=l.item()

            optimizer.zero_grad()
            scaler.scale(l).backward() # loss is scaled to prevent underflow issues
            scaler.step(optimizer)
            scaler.update()
        print("Epoch", epoch+prev_epochs,  ":", train_loss/len(trainloader.dataset))   
    end= time()
    print("\nAvg. time per epoch:", (end-st)/epochs)
    print("\nTotal number of epochs:", epochs + prev_epochs)
        
    if opt.partial:
        tot_epoch = epochs + prev_epochs
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': tot_epoch
        }, "checkpoint_" + str(tot_epoch) + "_" + str(opt.level))
    else:
        torch.save(model.state_dict(), opt.modelName)


