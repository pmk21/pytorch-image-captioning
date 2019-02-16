import os
import torch
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_train_ids(filename):
    """
    Loads ids of images that need to be trained.
    """
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()
    img_ids = [i.split('.')[0] for i in lines]
    return img_ids

def load_descriptions(filename):
    """
    Loads all descriptions along with image ids.
    """
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()
    vocab = set()
    desc = dict()
    for i in lines:
        t = i.split()
        img_id, img_desc = t[0], t[1:]
        vocab.update(img_desc)
        img_desc = ' '.join(img_desc)
        img_desc = img_desc.rstrip('\n')
        img_desc = 'startseq ' + img_desc + ' endseq'
        if img_id not in desc:
            desc[img_id] = []
        desc[img_id].append(img_desc)
    
    return desc, len(vocab)

def max_len(lines):
	return max(len(d[0].split()) for d in lines)

def create_tokenizer(descriptions):
    """
    Creates a keras tokenizer object which helps us convert strings
    to sequences later.
    """
    lines = [i[0] for i in descriptions.values()]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    #Store the dictionary of string to integer mappings
    with open("word_index.txt", "wb") as myFile:
        pickle.dump(tokenizer.word_index, myFile)
    
    return tokenizer

class Flickr8kDataset(Dataset):
    def __init__(self, train_file, desc_file, root_dir, transform=None):
        """
            Args:
            train_file (string): Path to the text file with train image ids.
            desc_file (string): Path to the text file with image ids and preprocessed descriptions.
            root_dir (string): Directory with all images.
            transform (callabe, optional): Optional transforms to be applied on a sample.
        """
        self.img_ids = load_train_ids(train_file)
        self.descriptions, self.vocab_size = load_descriptions(desc_file)
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = create_tokenizer(self.descriptions)
        #self.max_length = max_len(list(self.descriptions.values()))

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_with_ext = self.img_ids[idx] + '.jpg' 
        img_name = os.path.join(self.root_dir, img_with_ext)
        img = Image.open(img_name)
        img = img.resize((128, 128))
        
        if self.transform is not None:
            img = self.transform(img)

        target = self.tokenizer.texts_to_sequences(self.descriptions[self.img_ids[idx]])[0]
        #Convert sequence to tensor
        target = torch.tensor(target)

        return img, target
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 224, 224).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 224, 224).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]      
    
    return images, targets, lengths

def get_loader(train_file, desc_file, root_dir, batch_size, shuffle, num_workers, transform):
    """
    Returns a torch.utils.data.Dataloader object for the Flickr8k Dataset
    """
    flickr8k = Flickr8kDataset(train_file,desc_file, root_dir, transform)
    data_loader = DataLoader(dataset=flickr8k, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader