"""
In this second example, we will take a closer look at the dataloader.

So step into the src/dataloader.py file and set a breakpoint in the load_data_wrapper function.
Run the code and step into the load_cat_dog_data function and see whether you can figure out what is happening.

What you see in this file is fairly standard in pytorch.
We have created a custom dataset class, inheriting from torch.utils.data.Dataset.
The dataset knows the amount of samples in the dataset and have some iterator responsible for fetching new samples as needed.
A dataloader is a wrapper around a dataset such that we can fetch not just one example at a time, but a whole batch of them.
The dataloader also provides a bunch of performance improvements that we won't get into now, but which you can read more on in the advanced section later.

A few questions to consider:

How do we get a sample out of a dataloader?
How do we know how many samples are in a dataloader?
Why are we returning 3 different dataloaders?
What happens if an image has both the name cat and dog in its filename?, what should happen?

Some more advanced questions:

When we extract a random subset of our data, (as we currently do in the dataloader), we will likely not end up with a completely balanced subset, is this a problem? how could you fix this?
Our current setup handles images of different sizes by using a transformation that crops them to the centermost 256x256 pixels. When could this be a problem, and what are some alternatives?
Currently, we do not have any data augmentation. What kind of data augmentation might we do on images?

In our example we used accuracy as a measure, but for unbalanced datasets this might not be a good measure.
   (imagine an unbalanced dataset with 99 cat images and 1 dog image.)
   by always predicting cat we will reach 99% accuracy on such a dataset.

   Alternatively our dataset could be balanced but the importance of different predictions could be skewed.
   For cancer screening for instance a false negative might mean that you miss that someone has cancer, whereas a false positive means that an unnecessary person gets additional testing.

   So depending on the problem and the importance of true positives, true negatives, false positives and false negatives, we have various measures that are usefull.
   For more information search google for: "f1-score, recall, precision, accuracy"

"""
from config.unet import Configuration
# from config.efficient_net_b2_pretrained import Configuration
from src.main import main

if __name__ == "__main__":
    conf = Configuration()
    loss = main(conf)



