import os
import torch
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def get_feature_vector(img):
    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    model = models.resnet34(pretrained=True)
    layer = model._modules.get('avgpool')
    model.eval()
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.view(512))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

def load_train_ids(filename):
    fp = open(filename, 'r')
    lines = fp.readlines()
    fp.close()
    img_ids = [i.split('.')[0] for i in lines]
    return img_ids

def load_descriptions(filename):
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
        if img_id not in desc:
            desc[img_id] = []
        desc[img_id].append(img_desc)
    
    return desc, len(vocab)

def max_len(lines):
	return max(len(d[0].split()) for d in lines)

def create_tokenizer(descriptions):
    lines = [i[0] for i in descriptions.values()]
    #print(lines[0:2])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
	# walk through each description for the image
    for desc in desc_list:
    	# encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
    	# split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
    		# split into input and output pair
        	in_seq, out_seq = seq[:i], seq[i]
    		# pad input sequence
        	in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
        	# encode output sequence
        	out_seq = to_categorical([out_seq], num_classes=vocab_size)[0] #Need to include vocabulary size
    		# store
        	X1.append(photo)
        	X2.append(in_seq)
        	y.append(out_seq)
    return np.vstack(X1), np.vstack(X2), np.vstack(y)

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
        self.max_length = max_len(self.descriptions.values())

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_with_ext = self.img_ids[idx] + '.jpg' 
        img_name = os.path.join(self.root_dir, img_with_ext)
        img = Image.open(img_name)
        #img = np.array(img)
        if self.transform:
            img = self.transform(img)
        feature_vect = get_feature_vector(img)
        in_img, in_seq, out_word = create_sequences(self.tokenizer, self.max_length, self.descriptions[self.img_ids[idx]], feature_vect, self.vocab_size)
        sample = {'image':img, 'feature_vector':in_img, 'in_seq':in_seq, 'out_word':out_word, 'caption':self.descriptions[self.img_ids[idx]]}
    
        return sample

flickr8k = Flickr8kDataset('Flickr8k_text/Flickr_8k.trainImages.txt', 'descriptions.txt', 'Flickr8k_Dataset')

for i in range(10):
    sample = flickr8k[i]
    #plt.imshow(sample['image'])
    #plt.show()
    #print(sample['in_seq'].shape, sample['out_word'].shape)
    print(sample["feature_vector"])
    break