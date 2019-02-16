import os
import torch
import numpy as np
import torchvision.transforms as transforms
from model import *
from data_loader import get_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
                    
batch_size = 16

data_loader = get_loader('Flickr8k_text/Flickr_8k.trainImages.txt', 'descriptions.txt', 'Flickr8k_Dataset', batch_size, True, 2, transform)

encoder = EncoderCNN(50).to(device)
decoder = DecoderRNN(50, 256, 4476, 1).to(device)

criterion = nn.CrossEntropyLoss()
params =  list(decoder.parameters()) #list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

total_step = len(data_loader)
log_step = 20
num_epochs = 1

for epoch in range(num_epochs):
    for i, (images, captions, lengths) in enumerate(data_loader):
        
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
        
        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        
        #Not training the encoder
        #encoder.zero_grad() 
        
        loss.backward()
        optimizer.step()

        # Print log info
        if i % log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                  .format(epoch+1, num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

torch.save(decoder.state_dict(), 'models/decoder_1.ckpt')