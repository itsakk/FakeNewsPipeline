import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image
from query_data import *

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        VGG = models.vgg19(pretrained=True)
        self.feature = VGG.features
        self.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        pretrained_dict = VGG.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
 
    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output
 
model = Encoder()
model = model.cuda()

 
def extractor(img, net, use_gpu):
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )
 
    img = transform(img)
 
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
 
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy().tolist()
    return y
        
def get_images_features(path, source, label, list_news_files):
    use_gpu = torch.cuda.is_available()

    images_features = []
    filenames = []    
    L = get_all_images(source, label, list_news_files)
    for i, tup in enumerate(L):
        try:
            feat = extractor(tup[0], model, use_gpu)
            images_features.append(feat)
            filenames.append(list_news_files[i])
        except:
            feat = [0]*4096
            images_features.append(feat)
            filenames.append(list_news_files[i])
            
    return pd.DataFrame({'image_feature': images_features, 'filename': filenames})