import torch
import torchvision

class WCT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
        
    def encode(self,x):
        return self.encoder(x)
    def decode(self,x):
        return self.decoder(x)
    
    def whitening(self, content):
        """content: encoded image to whiten"""
        fc = content.clone().detach()
        dims = list(fc.shape)
        N = fc.shape[-1] # not in the paper
        fc -= fc.mean(dim=1)[:,None]
        cov = torch.mm(fc,fc.T) /(N-1) # not in the paper, division by N
        E, D, _ = torch.linalg.svd(cov)
        D = torch.diag(D**-.5)
        fhatc = torch.mm(E,D).mm(E.T).mm(fc)
        return fhatc, dims
    
    def coloring(self, style,fc):
        """ style: encoded style image to feed the autoencoder
            fc: whitened content image, output of the whitening method"""
        fs = style.clone().detach()
        N = fs.shape[-1]
        mus = fs.mean(dim=1)[:,None]
        fs -= mus
        cov = torch.mm(fs,fs.T) / (N-1) # not in the paper
        E, D, _ = torch.linalg.svd(cov)
        D = torch.diag(D**.5)
        fhatcs = torch.mm(E,D).mm(E.T).mm(fc)
        fhatcs += mus
        return fhatcs
    
    def WCT(self,content,style,alpha=1):
        """ Apply Whitening/Coloring transformation.
            content: tensor of the image used as base
            style: tensor of the image from which retrieve the style to transfer 
            alpha: [0,1] blend between original content and stylized-content"""
        with torch.no_grad():
            content = self.encode(content.reshape([1,3,224,224]))
            style = self.encode(style.reshape([1,3,224,224]))
            dim_original = content.shape[2]
            fc = torch.flatten(content[0],1)
            fhatc, dims = self.whitening(fc)
            fs = torch.flatten(style[0],1)
            transformed = self.coloring(fs,fhatc)
            if(alpha != 1):
                transformed = alpha*transformed + (1-alpha)*fc
            transformed = transformed.unflatten(1,[dim_original,dim_original])
            dim_original = list(transformed.shape)
            transformed = self.decode(transformed.reshape([1]+ dim_original))
            return transformed[0]
        
class UST_Net(WCT):
    def __init__(self, level:int):
        """Initialize the model.
        - level: initializes the relative autoencoder presented in the paper."""
        super().__init__()
        if level == 1:
            self.encoder = torchvision.models.vgg19(weights = torchvision.models.VGG19_Weights.DEFAULT).features[:2]
            self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        
        if level == 2:
            self.encoder = torchvision.models.vgg19(weights = torchvision.models.VGG19_Weights.DEFAULT).features[:7]
            self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        if level == 3:
            self.encoder = torchvision.models.vgg19(weights = torchvision.models.VGG19_Weights.DEFAULT).features[:12]
            self.decoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels= 128,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.UpsamplingNearest2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.UpsamplingNearest2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        if level == 4:
            self.encoder = torchvision.models.vgg19(weights = torchvision.models.VGG19_Weights.DEFAULT).features[:21]
            self.decoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels= 256,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.UpsamplingNearest2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=256, out_channels= 256,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels= 256,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels= 256,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels= 128,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.UpsamplingNearest2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.UpsamplingNearest2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
        if level == 5:
            self.encoder = torchvision.models.vgg19(weights = torchvision.models.VGG19_Weights.DEFAULT).features[:30]
            self.decoder = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels= 512,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.UpsamplingNearest2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=512, out_channels= 512,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels= 512,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels= 512,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels= 256,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.UpsamplingNearest2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=256, out_channels= 256,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels= 256,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels= 256,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=256, out_channels= 128,kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.UpsamplingNearest2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.UpsamplingNearest2d(scale_factor=2),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), 
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
