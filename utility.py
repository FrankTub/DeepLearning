# Due to some errors I had to change the background of matplotlib to get it to work in this terminal, this code is not necessary in git bash!
import matplotlib
matplotlib.use('Agg')
# to plot some nice figures
import matplotlib.pyplot as plt
# to make some barplot
import seaborn as sns

import os

# Used for loading the data from a directory structure
# [Pretrained models](https://pytorch.org/docs/master/torchvision/models.html)
from torchvision import transforms, datasets

# Used to create dataloader object
import torch

# numpy is awesome
import numpy as np

# To process a single image and make a prediction on this single image
from PIL import Image

def get_data_loader(data_dir):

    # Name used for the training data
    trainkey = 'train'
    # Name used for validation data
    validkey = 'valid'
    # Name used for testing data
    testkey  = 'test'

    # Do some prechecks to make sure we are creating a valid Directory
    if not data_dir.endswith(os.sep):
        homedir = data_dir + os.sep
    else:
        homedir = data_dir
    train_dir = homedir + trainkey
    valid_dir = homedir + validkey
    test_dir  = homedir + testkey

    # Making sure that all these directories exist
    if not os.path.isdir(train_dir):
        print("{} is not a valid training directory".format(train_dir))
        return None
    if not os.path.isdir(valid_dir):
        print("{} is not a valid training directory".format(valid_dir))
        return None
    if not os.path.isdir(test_dir):
        print("{} is not a valid training directory".format(test_dir))
        return None

    # Create dictionary for the different directories that we have
    dirs =  { trainkey : train_dir
            , validkey : valid_dir
            , testkey  : test_dir}

    # Create a dictionary to store all transformations
    data_transforms = {
        trainkey: transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        # Note that we do not want randomresizedcrop here, we just want our resized and centercropped image
        validkey: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        # Note that we do not want randomresizedcrop here, we just want our resized and centercropped image
        testkey : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # now creating our datasets is just a one-liner, this is done with ImageFolder, again just a one-liner (yeah I know it's spread in two lines, but still:))
    img_datasets  = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x])
                     for x in [trainkey, validkey, testkey]}

    # Is usefull in the do_training function
    dataset_sizes = {x: len(img_datasets[x])
                     for x in [trainkey, validkey, testkey]}

    # Create our dataloader object, again just a one-liner (yeah I know it's spread in two lines, but still:))
    dataloaders   = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size=16, shuffle=True)
                     for x in [trainkey, validkey, testkey]}

    # Store the number of features that we are trying to predict in a variable.
    clf_output = len(img_datasets[trainkey].classes)

    return dataloaders, img_datasets, clf_output

def predict(image_path, model, device, topk=5, cat_to_name=None):
    ###########################################################################################################
    # Function predict
    # Input parameters
    #    - image_path     : Relative/absolute filepath of image.
    #                       Eg '/home/workspace/aipnd-project/flowers/test/28/image_05230.jpg'
    #    - model          : Neural network model
    #    - topk           : the top k largest probabilities that the model predicts, by default this value is
    #                       set to 5.
    # Returns
    #    - topk_prob      : the topk highest values of probabilites
    #    - topk_flower    : the corresponding flower names
    #
    # Predict the class (or classes) of a single image using a trained deep learning model. Note that I chose
    # to return only the probabilities and the flower name. Not the corresponding value that corresponds to a
    # flower. This is in my opinion what this function should do.
    ###########################################################################################################

    # The model needs to be in evualate mode, we don't want to train it on accident and use dropout xD
    model.eval()

    # Process image using the function that I have created
    img = process_image(image_path)

    # Now we need to convert our numpy array back to a Tensor. Note that the type of tensor may vary
    # on if we want to execute the code on cpu or on cuda
    img = torch.from_numpy(img).type(torch.FloatTensor)

    # If we want to execute the code on cuda then we need to cast the FloatTensor to a cuda variant
    if device == 'cuda:0':
        img = img.cuda()

    # The model expects the batchsize as the first element, it was suggested by josh to use unsqueeze_
    img = img.unsqueeze_(0)

    # Now can use the image as input in our model, we need to convert the outcome of the model to the
    #probabilities using torch.exp. Since we are doing predictions we don't need the gradients.
    with torch.no_grad() :
        probabilities = torch.exp(model.forward(img))

    # With topk function of tensors we get back the highest values and the index of these highest values
    topk_prob, topk_idx = probabilities.topk(topk)

    # Convert tensor to list
    topk_prob = topk_prob.tolist()[0]
    topk_idx  = topk_idx.tolist()[0]

    # Convert keys and values in the dictionary that we already have. Now for each index we now to which class value it belong
    # Note that this is a dictionary therefore surrounded by {}
    idx_to_class = {y:x for x,y in model.class_to_idx.items()}

    # From the model we got the index back, based on this index we want the retrieve the corresponding value
    # Now we have the value we can make the translation to flower name based on our json file, we already have this
    # json file in memory in the variable cat_to_name
    if cat_to_name :
        topk_flower = [cat_to_name[idx_to_class[cls]] for cls in topk_idx]
    else :
        topk_flower = [idx_to_class[cls] for cls in topk_idx]

    return topk_prob, topk_flower

def process_image(image_path):
    ###########################################################################################################
    # Function process_image
    # Input parameters
    #    - image_path     : Relative/absolute filepath of image.
    #                       Eg '/home/workspace/aipnd-project/flowers/test/28/image_05230.jpg'
    # Returns
    #    - numpy array    : converted image to a numpy array
    #
    # Scales, crops, and normalizes a PIL image for a PyTorch model,
    # returns an Numpy array
    ###########################################################################################################

    # First create a PIL object to do some transformations.
    pil_img = Image.open(image_path)

    # Resize the longest side (width/height) to 256, preservering the ratio
    img = transforms.Resize(256)(pil_img)

    # Now we need to do some cropping, we need the following logic to keep an image of 224 by 224 pixels
    lmargin = (img.width-224) / 2   # left
    bmargin = (img.height-224) / 2  # bottom
    rmargin = lmargin + 224         # right
    tmargin = bmargin + 224         # top

    img = img.crop((lmargin, bmargin, rmargin, tmargin))

    # Now it is time to convert our pil image to a numpy array.
    img = np.array(img) / 255

    # Do some normalization on the image
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std

    # Make sure that color is the first dimension, this is what our model is expecting
    img = img.transpose((2, 0, 1))

    return img

def imshow(image, ax=None, title=None):
    ###########################################################################################################
    # Function imshow
    # Input parameters
    #    - image          : A processed image that is being returned from function process_image
    #    - ax             : Axes of matplotlib
    #    - title          : Title of the flower that is being displayed
    #
    # Displays an image that was cropped and resized by process_image
    ###########################################################################################################
    if ax is None:
        fig, ax = plt.subplots()

    # Add a title to the plot if this is given in the function call
    if title:
        plt.title(title)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes it is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing and normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def show_prediction(image_path, model, device, topk, cat_to_name=None):
    ###########################################################################################################
    # Function show_prediction
    # Input parameters
    #    - image_path     : Relative/absolute filepath of image.
    #                       Eg '/home/workspace/aipnd-project/flowers/test/28/image_05230.jpg'
    #    - model          : Neural network model
    #
    # Returns
    #    - pyplot object with two subplots. One plot containing the image of the flower and the other subplot
    #      containing a bargraph with probabilities of the topk flowers
    #
    # Visualize the prediction of the model with an image of the flower including the name of the flower along
    # with it. Make a barplot of the topk most likely flowers and their probabilities
    ###########################################################################################################
    # Set up plot, we create two plots, above eachother

    plt.figure(figsize = (6,8))

    ax = plt.subplot(2,1,1)


    # Now do some trick to get the label for the corresponding flower
    # WARNING: this assumes that the first integer that is a directory name in our image_path is the label!!
    title = None
    flow = None
    for i in image_path.split('/') :
        if i.isdigit() :
            # We found the first digit, now store the flower name in the flow variable. Since we assume that it is always the
            # first directory name we can break our for loop
            flow = i
            break

    # Only if we pass a variable in this cat_to_name should we try to map which flower it is
    if cat_to_name :
        if flow :
            title = cat_to_name[i]
    else :
        title = flow

    # Plot flower
    img = process_image(image_path)
    imshow(img, ax, title = title)
    # Make our predictions using the model
    probs, flowers = predict(image_path, model, device, topk, cat_to_name)
    # Plot bar chart

    plt.subplot(2,1,2)
    sns.barplot(x=flowers, y=probs, color=sns.color_palette()[0]);

    return plt
