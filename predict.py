# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

load_checkpoint('checkpoint.pth')  
print(model)


def process_image(image):
    
    img = Image.open(f'{image}' + '.jpg')

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    img_transform = transform(img)
    
    img_array = np.array(img_transform)
    
    return img_array


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

image_x = data_dir + '/test' + '/1/' + 'image_06743'
img_y =  process_image(image_x)

imshow(img_y, ax=None, title=None)

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file   
    
    loaded_model = load_checkpoint(model).cpu()
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    img_dimension = img_tensor.unsqueeze_(0)

    loaded_model.eval()
    
    with torch.no_grad():
        output = loaded_model.forward(img_dimension)

    probabilities = torch.exp(output)
    probabilities_top = probabilities.topk(topk)[0]
    index_top = probabilities.topk(topk)[1]
    
    probabilities_top = np.array(probabilities_top)[0]
    index_top = np.array(index_top[0])
    
    class_to_idx = loaded_model.class_to_idx
    
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    classes_top = []
    for index in index_top:
        classes_top += [indx_to_class[index]]
        
    return probabilities_top, classes_top


model_path = 'checkpoint.pth' 
image_path = data_dir + '/test' + '/11/' + 'image_03141'

probabilities,classes = predict(image_path, model_path, topk=5)

print(probabilities)
print(classes)

# TODO: Display an image along with the top 5 classes

model_path = 'checkpoint.pth'
image_path = data_dir + '/test' + '/11/' + 'image_03141'

probabilities,classes = predict(image_path, model_path, topk=5)

names = []
for i in classes:
    names += [cat_to_name[i]]

image = Image.open(image_path+'.jpg')

f, ax = plt.subplots(2,figsize = (8,12))

ax[0].imshow(image)
ax[0].set_title(names[0])

y_names = np.arange(len(names))
ax[1].barh(y_names, probabilities, color='green')
ax[1].set_yticks(y_names)
ax[1].set_yticklabels(names)
ax[1].invert_yaxis() 

plt.show()

