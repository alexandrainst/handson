"""
In this fourth example, we will take a closer look at the model optimization.

At first the optimization might seem a bit complicated, but the core of the optimization is actually rather simple.
What makes it looks complicated is all the reporting and tracking of loss and accuracy.
Try to step through the optimization code and see whether you can understand the various parts of the code.

Here we will briefly talk about some of the design choices that are behind this pipeline.

1) Typically you want to train a model many times and easily be able to compare current runs to previous runs and often on these runs are done on a remote server.
    Our current implementation does not really offer this.
    To get something like this we suggest the usage of some reporting framework like mlflow, tensorboard, or other such tools that automatically logs metrics and hyperparameters and allows easy visualization and comparison through an API that you can access remotely.
    This also allows your entire team to run various models on various computers and compare them all in a central API.

2) Currently, we do not use the validation dataset for anything, but typically you would want to use this to find the best model.

3) In a more realistic pipeline we also need to be able to save and load the model.
    Saving and loading models is something that needs to fit into the overall framework you run your machine learning models in.
    For information on how saving and loading models might be done see:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html

A few key questions to consider:

As mentioned, the functions in optimizer.py have a lot of reporting information that are not essential for the training of a model.
    Try to make a copy of the two functions in optimizer.py and remove all the reporting information and make the code as simple as possible.
    Can you train your model with these new functions?

At each epoch of the optimization, the model goes through the training dataset followed by the validation dataset.
    As previously stated, the purpose of the validation dataset is to help find the best model, how should the validation dataset help with this? (How would you incorporate that into your optimization function?)
How does the validation dataset differ from the test dataset?
How come the validation loss is lower than the training loss?

"""
from config.unet import Configuration
# from config.efficient_net_b2_pretrained import Configuration
from src.main import main

if __name__ == "__main__":
    conf = Configuration()
    loss = main(conf)



