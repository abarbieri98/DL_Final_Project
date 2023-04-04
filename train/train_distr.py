import sys
sys.path.insert(1,"../")
import model as m
import torch
import torchvision
import numpy as np
import argparse
from time import time
import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# The following code is the same as the one present inside the "train.py", it was just 
# modify to allow training on multiple GPUs.
# Path and other parameters may differ because this script was only ran inside ORFEO.

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank: int, world_size: int, opt):
    ddp_setup(rank, world_size)

    path_data = "../../COCO_10000"
    train_test = [8000, 1999]

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(256,256)),
        torchvision.transforms.CenterCrop(size=(224,224)),
        torchvision.transforms.ToTensor(),
        ])
    data = torchvision.datasets.ImageFolder(root=path_data, transform=data_transform) 
    train,_ = torch.utils.data.random_split(data,train_test, generator=torch.Generator().manual_seed(42)) 
    np.random.seed(42)
    if opt.subset > 0:
        train_subset = int(np.ceil(opt.subset * .8))    
        idx_train = np.random.choice(np.arange(0,stop = train_test[0]),train_subset)
        subtrain = torch.utils.data.Subset(train,idx_train)
        trainloader = torch.utils.data.DataLoader(subtrain, batch_size=opt.batchSize, pin_memory = True, shuffle=False,
            sampler=DistributedSampler(subtrain))
    else:
        trainloader = torch.utils.data.DataLoader(train, batch_size=opt.batchSize, shuffle=False,
            sampler=DistributedSampler(train))
        
    model = m.UST_Net(level=opt.level)
    model=model.to(rank)
    model=DDP(model, device_ids=[rank], find_unused_parameters=True)


    for param in model.module.encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scaler = torch.cuda.amp.GradScaler()
    loss = torch.nn.MSELoss(reduction="sum")
    epochs = opt.epoch
    lam = opt.lam
    prev_epochs = 0
    if opt.loadCheckpoint:
        prev_check = torch.load(opt.loadCheckpoint)
        model.module.load_state_dict(prev_check['model_state_dict'])
        optimizer.load_state_dict(prev_check['optimizer_state_dict'])
        prev_epochs = prev_check['epoch']

    st = time()
    for epoch in np.arange(epochs):
        print(f"[GPU{rank}]")
        model.train()
        train_loss = 0
        for x,_ in iter(trainloader):
            x=x.to(rank)
            x_hat=model.forward(x)
            fx = model.module.encode(x)
            fx_hat = model.module.encode(x_hat)
            with torch.autocast(device_type = "cuda", dtype = torch.float16):
                l=loss(x_hat,x) + lam*loss(fx,fx_hat)
                train_loss+=l.item()

            optimizer.zero_grad()
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
        print("Epoch", epoch+prev_epochs,  ":", train_loss/len(trainloader.dataset))   

    end= time()
    print("\nAvg. time per epoch:", (end-st)/epochs)
    print("\nTotal number of epochs:", epochs + prev_epochs)

    if rank==0: 
        if opt.partial:
            tot_epoch = epochs + prev_epochs
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': tot_epoch
            }, "checkpoint_" + str(tot_epoch) + "_" + str(opt.level))
        else:
            torch.save(model.module.state_dict(), opt.modelName)

    destroy_process_group()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchSize", type=int,default=128, help='batch size')
    parser.add_argument("--epoch", type=int,default=500, help='number of epochs for the training')
    parser.add_argument("--lr", type=float,default=1e-4, help='learning rate')
    parser.add_argument("--modelName", type = str, default = "another_model", help= "File name for the trained model (usually [class of the model]_[n epochs])")
    parser.add_argument("--lam", type = float, default = 1, help= "Lambda parameter for loss function")
    parser.add_argument("--subset", type = int, default = 9999, help= "Subset dataset for training (0 for full dataset)")
    parser.add_argument("--level", type=int, default=5, help="Select the the depth of the model.")
    parser.add_argument("--partial", action='store_true', default=False, help="Activate partial training, saving the state of the model and optimizer at the end.")
    parser.add_argument("--loadCheckpoint", type= str, default="", help="Checkpoint name to load")

    opt = parser.parse_args()

    world_size=torch.cuda.device_count()

    mp.spawn(main, args=(world_size, opt), nprocs=world_size)


    


