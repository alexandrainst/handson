"""
In this third example, we will take a closer look at the model.

There exist a wealth of different models all with various strengths and weaknesses.

Diving into model details is beyond the scope of this workshop,
but suffice to say that model architecture is constantly evolving and unless you are an expert you are likely best off just using some well performing neural network for a problem similar to the one you are interested in.

Common to all neural networks is that they consist of some amount of trainable parameters and in general the more parameters a model has, the more memory it requires to train/run and the longer it takes.

In this project we have provided a simple U-net convolutional model, as well as a few models from torchvisions modelhub.

Set a breakpoint in model_loader_wrapper function and run the code to there.
step into the instantiation of the Unet() and take a look around.
Next put a breakpoint in the forward function of the UNet Class, and run your code until you hit that breakpoint.
Look at the shape of the data as it goes through the forward function of the model.


A few key questions to consider:

Based on the shape of the data as it changed through the forward function of the model what do you think is happening?
When we use a pretrained model and change the last layers to fit our particular needs, do we then need to retrain the whole model?

"""
from config.unet import Configuration
# from config.efficient_net_b2_pretrained import Configuration
from src.main import main

if __name__ == "__main__":
    conf = Configuration()
    loss = main(conf)



