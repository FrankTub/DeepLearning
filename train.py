###########################################################################################################
# Script train.py
# Example of use:
#    - python train.py data_directory              ==> where data_directory is directory where the train data is located.
#    - python train.py data_dir -s save_directory  ==> -s allows you to dictate the directory where the checkpoint will be saved.
#    - python train.py data_dir -a "vgg13"         ==> -a dictates the architecture of the pretrained model.
#    - python train.py data_dir -g                 ==> -g enables GPU training.
#    - python train.py data_dir -L 0.001           ==> -L sets the learning rate.
#    - python train.py data_dir -H 512             ==> -H sets the number of hidden_layers in the classifier.
#    - python train.py data_dir -E 4               ==> -E sets the number of iteration over the training set.
#
#
# Script to train your neural network from the command line. See python train.py -h for more help.
###########################################################################################################

# To be able to give optional arguments into this training program
import argparse
# To validate if the directories given are existing
import os
# Import all model functions necessary to do the things we need to do.
import model as mo
# Check if cuda is available
import torch
# Variable to store the filename prefix, very ugly solution..
ckp_fileprefix = "checkpoint_"
# Import all utility functions necessary to do the things we need to do.
import utility as ut

# The problem with this is that it still does not check if your default directory is valid.
# It just assumes that your default directory is valid. So make sure this is the case!
class readable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


parser = argparse.ArgumentParser( description="Program to train your neural network to the max:) Note that when you pass in a combination of save_dir and architecture that already exist, that the options --epoch and --hidden_layers are ignored. These options can only be set for a new model."
                                , add_help=True
                                , formatter_class=argparse.MetavarTypeHelpFormatter
                                )
# The directory where the image data is stored is required.
parser.add_argument( 'directory'
                   , metavar='Directory'
                   , type=str
                   , action=readable_dir
                   , help = 'Directory where the image data can be found. Note that this is probably an environment specific location!!'
                   )

# Add an option to enable the user to train the model on GPU, this is just
parser.add_argument( '-g'
                   ,'--GPU'
                   , dest = 'GPU'
                   , action="store_true"
                   , default=False
                   , help="Gives you the option to train the model on GPU, by default this is set to False and therefore training will be done on CPU."
                   )

# Add an option to be able to set the directory where we save the checkpoint
# Note that the default value in this case should be of type list instead of string
# Because in args.checkpoint_dir we will get a list if we give -s checkpoint_dir
parser.add_argument( '-s'
                   , '--save_dir'
                   , type=str
#                    , default='C:/Users/A696260/Documents/Python Scripts/Deep Learning/'
                   , action=readable_dir
                   , dest="checkpoint_dir"
                   , help="Gives you the ability to determine in which directory your checkpoint will be saved. If there is already a checkpoint in this directory for the architecture that you want to use it will load that checkpoint. Note that this is probably an environment specific location!!"
                   )

parser.add_argument( '-a'
                   , '--arch'
                   , type=str
                   , action="store"
                   , default='vgg19'
                   , dest="architecture"
                   , help="Gives you the ability to chose the pretrained neural network you want to train. By default the pretrained model is vgg19"
                   )

parser.add_argument( '-L'
                   , '--learning_rate'
                   , type=float
                   , action="store"
                   , default=0.001
                   , dest="learning_rate"
                   , help="Gives you the ability to set the learning rate of the optimizer"
                   )

parser.add_argument( '-H'
                   , '--hidden_units'
                   , type=int
                   , action="store"
                   , default=4096
                   , dest="hidden_units"
                   , help="Gives you the ability to set the number of hidden units in the classifier of the model"
                   )

parser.add_argument( '-E'
                   , '--epochs'
                   , type=int
                   , action="store"
                   , default=2
                   , dest="epochs"
                   , help="Gives you the ability to set the number of iterations over the training data set"
                   )

# Add a version number to this program
parser.add_argument( '-v'
                   , '--version'
                   , action='version'
                   , version='%(prog)s 1.00.00')

args = parser.parse_args()

# First determine if this piece of code has to be executed on GPU or not.
if args.GPU :
    # We know we want to the execute this on cuda. But maybe cuda is not available..
    if torch.cuda.is_available():
        device = "cuda:0"
    else : # So if cuda is not available don't do some unexpected things, just raise an error.
        raise ValueError("We wanted to execute this training on GPU, but cuda is not available!!\nPlease remove the -g option or make sure cuda is available.")
else :
    device = 'cpu'

print("The training is done on {}".format(device))

if args.checkpoint_dir :
    ckp_filepath = args.checkpoint_dir + ckp_fileprefix + args.architecture + ".pth"
else :
    ckp_filepath = ckp_fileprefix + args.architecture + ".pth"

if os.path.isfile(ckp_filepath) :
    print("Checkpoint {} recognized, continue training this model!".format(ckp_filepath))
    model = mo.load_checkpoint(ckp_filepath, device)
else :
    print("Checkpoint {} not recognized, starting from scratch!".format(ckp_filepath))
    model = mo.init_model(args.directory, args.architecture, args.learning_rate, args.hidden_units)

# Create an object where we can iterate over the data
dataloaders, img_datasets, _ = ut.get_data_loader(args.directory)

# Is usefull in the do_training function
dataset_sizes = {x: len(img_datasets[x])
                 for x in ['train', 'valid', 'test']}
# Now we are ready to do some training
model = mo.do_training(model, dataloaders, dataset_sizes, device, epochs = args.epochs)
# Done with the fucking training, now save the network again
mo.save_checkpoint(model, args.architecture, img_datasets['train'], ckp_filepath)
