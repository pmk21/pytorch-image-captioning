import string

def load_descriptions(filepath, caption_id):
    fp = open(filepath, 'r')
    #Read all the captions
    text = fp.readlines()
    descriptions = []

    for i in text:
        if i.split()[0][-1:] == str(caption_id): #Get all the descriptions matching the caption id
            descriptions.append(i)

    print("Number of descriptions loaded: {0}".format(len(descriptions)))
    fp.close()
    return descriptions

def map_descriptions(descriptions):
    """
        Create a mapping from the image id to the corresponding caption.
    """
    mapping = dict()
    for desc in descriptions:
        tokens = desc.split()
        image_id, image_cap = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0] #Removing the file extension
        image_cap = ' '.join(image_cap)
        
        if image_id not in mapping:
        	mapping[image_id] = list()
        
        mapping[image_id].append(image_cap)
	
    return mapping

def clean_descriptions(descriptions):
    """
        Convert the descriptions to lower case, remove punctuation, remove hanging
        's' and 'a' and remove tokens with numbers.
    """
    #Prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    for _, desc_list in descriptions.items():
        desc = desc_list[0]
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [w.translate(table) for w in desc]
        desc = [word for word in desc if len(word)>1]
        desc = [word for word in desc if word.isalpha()]
        desc_list[0] =  ' '.join(desc)

def to_vocabulary(descriptions):
    """
        Build a set of all description words
    """
    all_desc = set()
    for key in descriptions.keys():
        d = descriptions[key][0]
        all_desc.update(d.split())
    return all_desc

def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		lines.append(key + ' ' + desc_list[0])
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

f_name = 'Flickr8k_text/Flickr8k.token.txt'
idx = 0
descriptions = load_descriptions(f_name, idx)
img_desc = map_descriptions(descriptions)
clean_descriptions(img_desc)
vocabulary = to_vocabulary(img_desc)
#print(list(img_desc.keys())[0], img_desc[list(img_desc.keys())[0]])
print('Vocabulary Size: %d' % len(vocabulary))
#Save descriptions to a text file
save_descriptions(img_desc, 'descriptions.txt')