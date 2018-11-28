# Used for loading the data from a directory structure
# [Pretrained models](https://pytorch.org/docs/master/torchvision/models.html)
from torchvision import transforms, datasets, models
# Used to create dataloader object
import torch
# Otherwise we cannot simply use nn.* ==> torch.nn.*
from torch import nn
# To make our own classifier, gives us cleaner code
from collections import OrderedDict
# To create optimizer object
from torch import optim
# To change the learning rate with the number of epochs
from torch.optim import lr_scheduler
# To make an unique copy of our model, not just a variable that points to the same object
import copy
# To check the performance
import time

import utility as ut

# To check progress of iteration
from tqdm import tqdm

def transfer_learning(architecture):
    ###########################################################################################################
    # Function transfer_learning
    # Input parameters
    #    - architecture   : Name of the pretrained model
    #
    # Define all pretrained model options in a single function and return model object
    ###########################################################################################################

    if   architecture == 'vgg16' :
        model = models.vgg16(pretrained=True)
    elif architecture == 'vgg19' :
        model = models.vgg19(pretrained=True)
    else :
        raise ValueError('Architecture {} is not recognized'.format(architecture))
    return model

def init_model(data_dir, architecture, learning_rate, hidden_units):
    model = transfer_learning(architecture)
    # Making sure that the pretrained model weights are not updated by our training
    for param in model.parameters():
        param.requires_grad = False

    clf_input = model.classifier[0].in_features
    *_, clf_output = ut.get_data_loader(data_dir)
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(clf_input, hidden_units)),
                              ('relu1', nn.ReLU()),
                              ('dp1',nn.Dropout(0.5)),
                              ('fc2', nn.Linear(hidden_units, clf_output)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    # Store the optimzizer, criterion and scheduler in the model
    model.criterion = nn.NLLLoss()
    model.optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.scheduler = lr_scheduler.StepLR(model.optimizer, step_size=4, gamma=0.1)
    return model


def save_checkpoint(model, architecture, image_dataset, filename):
    ###########################################################################################################
    # Function save_checkpoint
    # Input parameters
    #    - model          : Neural network model
    #    - architecture   : Name of the pretrained model
    #    - image_dataset  : Dataloader object that contains only data for a specific set we want to do the evaluation on
    #    - filename       : Relative/absolute filepath of checkpoint. Eg 'classifier.pth'
    #
    # Function to store the current status of the classifier in a checkpoint, so that it can be reloaded in
    # another session. Note that we don't store the complete model. Just the classifier state_dict. This can be
    # done because we don't change the base model, just the classifier. This function only works under the assumption
    # that the classifier contains 3 layers ==> [input|hidden|layer]
    ###########################################################################################################
    model.class_to_idx = image_dataset.class_to_idx

    # Dictionary object that contains the number of [input|hidden|output] layers of the classifier
    dim_clf = {'input'  : model.classifier[0].in_features,
               'hidden' : model.classifier[0].out_features,
               'output' : model.classifier[3].out_features}

    model.cpu()
    torch.save({'architecture'     : architecture,
                'state_dict'       : model.classifier.state_dict(),
                'class_to_idx'     : model.class_to_idx,
                'optim_state_dict' : model.optimizer.state_dict(),
                'sched_state_dict' : model.scheduler.state_dict(),
                'dim_clf'          : dim_clf},
                filename)

def load_checkpoint(filepath, device):
    ###########################################################################################################
    # Function load_checkpoint
    # Input parameters
    #    - filepath       : Relative or absolute filepath of checkpoint
    #    - device         : where to execute the code on, options are [cpu|cuda] in the udacity classroom
    #
    # Returns
    #    - model          : Neural network object
    #
    # Function to load the last status of the classifier from a checkpoint
    ###########################################################################################################

    #Load the checkpoint
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    # Note that we did not store the complete model in the previous exercise, so that one we need to import as well
    model = transfer_learning(checkpoint['architecture'])

    # Making sure that we freeze the pretrained model weights, otherwise things will go south very soon.
    for param in model.parameters():
        param.requires_grad = False

    dim_clf = checkpoint['dim_clf']
    # Now we want to rebuild the classifier, but first we need to initialize it
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(dim_clf['input'], dim_clf['hidden'])),
                          ('relu1', nn.ReLU()),
                          ('dp1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(dim_clf['hidden'], dim_clf['output'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


    model.classifier = classifier
    model.classifier.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # We did not store this but making sure that we have the criterion in the model object.
    model.criterion = nn.NLLLoss()

    # Initially I thought I would have to put this statements just before returning the model, however this results in
    # errors for the optimizer that is stored in the model. So first put the model on cuda, and then add the optimizer
    # and scheduler.
    model.to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.optimizer = optimizer
    model.optimizer.load_state_dict(checkpoint['optim_state_dict'])

    scheduler = lr_scheduler.StepLR(model.optimizer, step_size=4, gamma=0.1)
    model.scheduler = scheduler
    model.scheduler.load_state_dict(checkpoint['sched_state_dict'])

    return model

def do_training(model, dataloaders, dataset_sizes, device, epochs = 4) :
    ###########################################################################################################
    # Function do_training
    # Input parameters
    #    - model       : neural network that will be trained
    #    - dataloaders : dataloader object that is dictionary with dataloader objects in it
    #    - device      : where to execute the code on, options are [cpu|cuda] in the udacity classroom
    #    - epochs      : number of iteration over the complete dataset
    #
    # Returns
    #    - model       : trained neural network model
    #
    # This function is used to train neural networks. Passing in the dataloader for a validation set allows us
    # to determine per iteration which model is performing the best. Per epoch, iteration over the dataset,
    # this validation is conducted. Only if the model is performing better than the previous
    #
    # The code was inspired by https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
    ###########################################################################################################

    # At First I tried to do the training on my local machine, however a single iteration with 64 images in it
    # took more than 8 minutes.. On local machine I have an Intel processor. Googled a lot but was not able to figure out
    # a solution to use the intel processor in a similar way as cuda is being used in the classroom. So then I just stuck
    # to the workspace provided by Udacity. I would like to know if there is an alternative for an intel processor to
    # use the GPU in python. Could not find any better option than pyopencl, which would require to write some code
    # ourselves. Does not seem like the correct option to me.
    # My local machine has a GPU: Intel(R) HD Graphics 5500.

    model.to(device)

    # Store the state_dict of the model in a variable, in case something goes wrong with the logic in this function
    best_model_sd = copy.deepcopy(model.state_dict())

    start = time.time()

    best_acc = 0

    for e in range(epochs) :

        # Printinig the epoch number
        print('Epoch {}/{}'.format(e + 1, epochs))
        print('-' * 11)

        # Per iteration we are doing some training and afterwards validation if we improved the model by training it so.
        for phase in ['train', 'valid'] :
            if phase == 'train' :
                # model needs to be put in training mode
                model.train()
                model.scheduler.step()
            else :
                # Then we are validating and don't want to train our model and use dropout etc
                model.eval()

            # Reset running values per iteration, initialize it as float so we don't run into problems later on
            running_loss     = 0.0
            running_corrects = 0.0

            # Very nice suggestion from a peer to use tqdm to keep track of the progress
            for inputs, labels in tqdm(dataloaders[phase]):
                # make sure the input and labels is in the correct device
                inputs, labels = inputs.to(device), labels.to(device)

                # Make sure that the gradient is reset to zero before we do any learning
                model.optimizer.zero_grad()

                # For validation we don't want the gradient, but for testing we do
                # So for testing we set it to true, and for validation we set it to false
                with torch.set_grad_enabled(phase == 'train') :
                    outputs = model.forward(inputs)
                    loss    = model.criterion(outputs, labels)

                # For training we want to the update our model
                if phase == 'train' :
                    loss.backward()
                    model.optimizer.step()

                # Calculate running loss for the phase, keep in account that the number of inputs is variable per phase
                running_loss += loss.item() * inputs.size(0)
                # Calculate predictions
                predictions = torch.max(outputs.data, 1)[1]
                # Calculate total number of correct predictions
                running_corrects += torch.sum(predictions == labels.data)

            # For this given phase we are done performing the iteration part. Now print some
            # details about our process
            epoch_loss = running_loss / dataset_sizes[phase]
            # Don't understand why in the bottom one we need to call double() on it, while on running_loss this is not necessary
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # If we do better on the validation set, we want to update the best accuracy variable and copy the state_dict
            # of the model into a new variable
            if phase == 'valid' and epoch_acc >= best_acc :
                best_acc = epoch_acc
                best_model_sd = copy.deepcopy(model.state_dict())
    print("Total duration was {:.0f}m {:.0f}s.\n Best accuracy was {}".format((time.time() - start) / 60, (time.time() - start) % 60, best_acc))

    model.load_state_dict(best_model_sd)
    return model

def do_evaluate(model, dataloader, device) :
    ###########################################################################################################
    # Function do_evaluate
    # Input parameters
    #    - model       : neural network that will be trained
    #    - dataloader  : dataloader object that contains only data for a specific set we want to do the evaluation on
    #    - device      : where to execute the code on, options are [cpu|cuda] in the udacity classroom
    #
    # Returns
    #    - accuracy    : Number of correctly identified elements divided by total number of elements
    #
    # This function is used to evaluate the neural network and check the accuracy.
    ###########################################################################################################

    # Use the agnostic code of one of the lessons to make sure it doesn't matter if we execute code on cpu/gpu
    model.to(device)
    # Must set the model in eval mode!!
    model.eval()

    correct = 0
    total = 0
    # For these computations we don't need the gradient. torch.no_grad() sets it only for these calculations to no gradient!
    with torch.no_grad():
        # Keep track of the status in the testloader
        for images, labels in tqdm(dataloader):
            # Use the agnostic code of one of the lessons to make sure it doesn't matter if we execute code on cpu/gpu
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            # Only interested in the accuracy here, in [0] are the probabilities and in [1] the indexes of the predictions
            predicted = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct / total)

def predict(image_path, model, topk=5):
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
    topk_flower = [cat_to_name[idx_to_class[cls]] for cls in topk_idx]

    return topk_prob, topk_flower
