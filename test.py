from PIL import Image
import os
import torch
import pickle
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
from model import *


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
parser.add_argument('--decoder_path', type=str, default='models/decoder-34.ckpt', help='path for trained decoder')
parser.add_argument('--vocab_path', type=str, default='word_index.txt', help='path to dictionary having str to idx mapping')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])

# Build models
encoder = EncoderCNN(50).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder = DecoderRNN(50, 256, 4476, 1).eval()
encoder = encoder.to(device)
decoder = decoder.to(device)

# Load the trained model parameters
decoder.load_state_dict(torch.load(args.decoder_path))


img_loc = args.image

# Prepare an image
image = load_image(img_loc, transform)
image_tensor = image.to(device)

# Generate an caption from the image
feature = encoder(image_tensor)
sampled_ids = decoder.sample(feature)
sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

# Convert word_ids to words
with open("word_index.txt", "rb") as myFile:    
    word_to_id = pickle.load(myFile) 

id_to_word = dict()
for i in word_to_id.keys():
    id_to_word[word_to_id[i]] = i

sampled_caption = []
for word_id in sampled_ids:
    word = id_to_word[word_id]
    sampled_caption.append(word)
    if word == 'endseq':
        break
sentence = ' '.join(sampled_caption)

# Print out the image and the generated caption
print (sentence)
image = Image.open(img_loc)
plt.imshow(np.asarray(image))
plt.show()