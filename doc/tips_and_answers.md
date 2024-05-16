# Debugging
1. Always start by training on 1 data sample and see that your model is able to match that completely. 
Then move to 1 batch of samples and see once again that you can fit that. 
If your model is unstable or does not train well on these simple tasks then fix those problems before moving on to anything more advanced.
2. Get things running on cpu before switching to gpu. If there is a problem on gpu, check whether it is also a problem on cpu. 
3. `with torch.autograd.detect_anomaly():` is useful for debugging faulty code. It will slow down you code quite significantly so remember to remove it once code is working.

# Caveat

- Unlike classical simulations where everything have to be 100% correct in order for you to get any sensible results, you can do things very wrong when it comes to ML and still get good results.
This can be both a blessing and a curse.
I once worked at a tech company, that was using some fairly advanced reversible neural networks, and after having worked there for a while I one day started to look closer at the neural network they were using and their models forward operation looked something like this:

      def forward(self, x):
          x1 = self.conv1(x)
          x2 = self.down1(x1)
          x3 = self.down2(x2)
          x4 = self.down3(x3)
          x5 = self.down4(x4)
          x4_up = self.up1(x4, x5)
          x3_up = self.up2(x3, x4_up)
          x2_up = self.up3(x2, x3_up)
          x1_up = self.up4(x1, x2_up)
          y = self.conv1(x1)
          y1 = self.avgpool(y)
          output = self.classifier(y1[:,:,0,0])
          return output

    The problem is the line
      
      y = self.conv1(x1)
    which should have been
  
      y = self.conv1(x1_up)
    As it is written it completely discards the majority of the network.
    Nevertheless, this network was used in production in this company and had been for months and no one had noticed this error.
    This brings me to the point: 
    Some small errors, like in classical simulation, will lead to utterly wrong results and nothing will be working.
    But other major errors will remove 75% of your calculations, and you might not even notice it because the neural networks can compensate for a lot of wrongs and will just silently try to do its best. (when we fixed the error the network did improve its results maybe 10%)

- Machine learning is a field that changes at incredible speeds. So things from just a few years ago could already be severely outdated.

# Solutions to questsions

## 0 overview

>The optimize_model step gives both a loss and an accuracy, what do you think these are? How do they differ?

Loss is the difference between the neural network predicted output and the ground truth according to the metric we defined in the loss function.
The accuracy is how many images the model classified correctly.

>There are 4 components (+configuration) that goes into the optimize_model, based on their names can you already imagine how they might work together?

The four components are: dataloader, optimizer, loss_fnc, and model.
Machine learning is similar to any other optimization problem (like linear regression with gradient decent). 
You have some data (dataloader), which you transform (model), you then compare the data to the desired result (loss_fnc). 
Finally the model is updated based on this loss (optimizer).


## 1 dataset

> How do we get a sample out of a dataloader?

The dataloader works as a generator, so you have to iterate over it to get the samples out.

> How do we know how many samples are in a dataloader?

len(dataloader.dataset), len(dataloader) will give you the amount of batches needed to iterate through the dataset.

>Why are we returning 3 different dataloaders?

It is standard practice in ml to have a training dataset, a validation dataset and a test dataset.
The training dataset is for training the model parameters. 
The validation dataset is for tuning the hyperparameters of the model (A hyperparameter is basically any free variable that is not part of the model weights - like for instance learning rate.) 
The test dataset is for comparing different models and should only be run once a model has been optimized over training and validation data. The test data is supposed to simulate realistic data that the model might be used on once training and deployed. 

>What happens if an image has both the name cat and dog in its filename?, what should happen?

At the moment it gets classified as a cat. What should happen is up to the developer, but likely we should be throwing a warning or exception of some kind. See ./doc/advanced/dataloader.py

>When we extract a random subset of our data, (as we currently do in the dataloader), we will likely not end up with a completely balanced subset, is this a problem? how could you fix this?

If the dataset is mostly balanced it is likely not something that needs to be done anything about. However, if the dataset is heavily imbalanced this needs to be accounted for. 
Two fixes could be to downsample the overrepresented categories, or upweight the underrepresented categories. See https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data for more info.   

>Our current setup handles images of different sizes by using a transformation that crops them to the centermost 256x256 pixels. When could this be a problem, and what are some alternatives?

If you have very large images then any crop to 256x256 will only give you a very small amount of the image, which might not even show the animal or only such a small part that it is impossible to determine the animal.
For a more general approach we should likely downsample/upsample the image to the desired resolution.

>Currently, we do not have any data augmentation. What kind of data augmentation might we do on images?

There are many data augmentations that one could use on an image, like rotation, translation, mirroring. Small changes in colors/intensities as well as noise could also be relevant. The idea is basically to use a small set of images to generate other images that still fall in the same category from which the model can learn more and therefore become more robust.

See ./doc/advanced/dataloader.py

## 2 model
>Based on the shape of the data as it changed through the forward function of the model what do you think is happening?

The model is a U-net, which is named so because it starts by downsampling the resolution in a number of steps, while increasing the number of channels. The downsampling essentially allows the corresponding convolutional filters to have a larger effective range. The reverse process is then happening in the later half of the U-net.

 
>When we use a pretrained model and change the last layers to fit our particular needs, do we then need to retrain the whole model?

It is a fairly standard technique when using a pretrained model, to only retrain the last layer of the model and keep all other layers frozen, this makes the fine-tuning even faster and usually gives good results. Depending on how similar the fine-tuning task is to the original task it might be better to allow retraining of all layers but with different learning rates for the different layers.  


## 3 optimization
>As mentioned, the functions in optimizer.py have a lot of reporting information that are not essential for the training of a model.
  Try to make a copy of the two functions in optimizer.py and remove all the reporting information and make the code as simple as possible.
  Can you train your model with these new functions?

See ./doc/advanced/optimizer.py

>At each epoch of the optimization, the model goes through the training dataset followed by the validation dataset.
    As previously stated, the purpose of the validation dataset is to help find the best model, how should the validation dataset help with this? (How would you incorporate that into your optimization function?)

The validation step helps us determine whether the model has overfit to the training data. See ./doc/advanced/optimizer.py

>How does the validation dataset differ from the test dataset?

The validation dataset is used to tune the hyperparameters and is considered a component in the overall optimization pipeline, whereas the test dataset is meant to emulate completely novel data and should only be used to evaluate the fully trained neural network.

>How come the validation loss is lower than the training loss?

Normally the validation loss is higher than the training loss, since the validation data are not directly being trained on whereas the training data are. However, this particular neural network includes [dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html), which is only active when the neural network is running in training mode. Without the dropout term we would expect the validation loss to be higher than the training loss.

## 4 debugging
>Try to find as many errors as possible

1. The learning rate is too high
2. We should not use the sigmoid activation function on the model output in this case
3. The loss function needs the output and the target
4. The backward() function is never called
5. The optimizer is never called, so the weights never update