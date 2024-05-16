"""
Before running this code, make sure you have downloaded the dataset needed. Run /data/download_cats_and_dogs.py to get the dataset.

All the examples in this folder will run the exact same code.
The only difference will be the comments here at the top detailing which parts of the code we will focus on.

In this first example. The goal is to get the code to run on your particular system, and to get a high overview of the code.

So start by running the code and see whether it is able to run successfully or whether it throws an error.
If the code runs successfully then great, you can move on to the second part, if not check the following:
- Did you remember to download the cats and dogs dataset? if not run /data/download_cats_and_dogs.py
- Did the code crash because it ran out of memory? In this case go to /examples/config/unet.py and set
    device: str = "cpu"
    If this still isn't enough you will need to lower the batch_sizes as well.
- Did the code crash because it cannot find/import the modules?
    check that you have installed the required packages (see pyproject.toml)
    ensure that you have the correct working directory (it should be {where-ever-your-repository-is}/dlpresentationmaterial/examples)


If the code ran without problem, then try to run it again in debugging mode.
Set a breakpoint on the first line of the code and step into the configuration and get a feel for the various parameters in there, do they all seem sensible?
Next step into the main function and go through the various steps it does. Do not step into any of the substeps of main, but feel free to inspect the various elements that the steps return.
Run the script to the end.


A few key questions to consider:

The optimize_model step gives both a loss and an accuracy, what do you think these are? How do they differ?
There are 4 components (+configuration) that goes into the optimize_model, based on what you have seen so far, can you already imagine how they might work together?

"""
from config.unet import Configuration
# from config.efficient_net_b2_pretrained import Configuration
from src.main import main

if __name__ == "__main__":
    conf = Configuration()
    loss = main(conf)



