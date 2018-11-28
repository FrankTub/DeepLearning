###########################################################################################################
# Script predict.py
# Example of use:
#    - python train.py input checkpoint            ==> where input is filepath of image and checkpoint is filepath where the model is saved.
#    - python train.py input checkpoint -t 4       ==> where 4 represents the number of flowers returned by this script.
#    - python train.py input checkpoint -c bla.json==> where bla.json represents the file that translates the number to a flower name.
#    - python train.py input checkpoint -g         ==> -g enables predictions on GPU.
#
# Script to use your neural network from the command line and make predictions on a single image. See python predict.py -h for more help.
###########################################################################################################
# to plot some nice figures
# import matplotlib.pyplot as plt
# To be able to give optional arguments into this training program
import argparse
# Check if cuda is available
import torch
# Import all model functions necessary to do the things we need to do.
import model as mo
# Import all utility functions necessary to do the things we need to do.
import utility as ut

import os.path

# To make translation from label to flower type
import json

# Added a class to make sure that if we put in some variables that it is ensured
# that the input is a valid readable file
class readable_file(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file=values
        if not os.path.isfile(prospective_file):
            raise argparse.ArgumentTypeError("readable_file:{0} is not a valid filepath".format(prospective_file))
        if os.access(prospective_file, os.R_OK):
            setattr(namespace,self.dest,prospective_file)
        else:
            raise argparse.ArgumentTypeError("readable_file:{0} is not a readable file".format(prospective_file))

parser = argparse.ArgumentParser( description="Script to make a prediction on your image using a trained neural network)"
                                , add_help=True
                                , formatter_class=argparse.MetavarTypeHelpFormatter
                                )

parser.add_argument( 'image_filepath'
                   , metavar='input'
                   , type=str
                   , action=readable_file
                   , help = 'Filepath of image which we want to make a prediction on.'
                   )

parser.add_argument( 'checkpoint'
                   , metavar='checkpoint'
                   , type=str
                   , action=readable_file
                   , help="Filepath of checkpoint"
                   )

# Add an option to enable the user to make a prediction using GPU
parser.add_argument( '-g'
                   ,'--GPU'
                   , dest = 'GPU'
                   , action="store_true"
                   , default=False
                   , help="Gives you the option to make prediction on GPU, by default this is set to False and therefore predicting will be done on CPU."
                   )

parser.add_argument( '-t'
                   , '--top_k'
                   , type=int
                   , action="store"
                   , default=5
                   , dest="top_k"
                   , help="Gives you the ability to set the number of flowers that are returned by this script. Sorted from most likely desc"
                   )

# json file linking an id to a flower, must be a valid file
parser.add_argument( '-c'
                   , metavar='--category_names'
                   , type=str
                   , action=readable_file
                   , default='cat_to_name.json'
                   , dest="category_names"
                   , help="Filepath of category_names json_file"
                   )

# Add a version number to this program
parser.add_argument( '-v'
                   , '--version'
                   , action='version'
                   , version='%(prog)s 1.00.00')

# Get an object to reference all our input arguments
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

cat_to_name = None
# Only if the default or supplied parameter for category_names is a file we will try to make a mapping from id to flower name
# This will ignore an invalid path to the json_file
if os.path.isfile(args.category_names) :
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

# Now we are ready to load the model
model = mo.load_checkpoint(args.checkpoint, device)

# Get a reference to the plot we want to make and save it
fig = ut.show_prediction(args.image_filepath, model, device, args.top_k, cat_to_name)

fig.savefig("pred_" + os.path.basename(args.image_filepath))
