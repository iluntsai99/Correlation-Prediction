# Correlation-Prediction

+ Training a CNN model to predict the correlation of scatter plots

+ You are provided with a dataset which contains images of scatter plots and correlations corresponding to each scatter plot. Given these images, predict the correlation between x and y values. To get acquainted with how it works, you can play the game here first: http://guessthecorrelation.com/

+ **Note: You do not need to conduct a detailed analysis for this task. Just training a model will suffice.**Follow the steps below to download the dataset and train a model.
    + Download the images and correlation values from this link: https://drive.google.com/file/d/1OqZJW8WUeUi2XrzJisJBHR4Er1lPowmG/view?usp=sharing 
    + The folder contains 150,000 images. Separate the images into training and test sets.
        + How many images will you use for training? How many images will you use for testing? Why?
            + **The ratio between train:test is 9:1. To be more specific, I used 135000 images for training and 15000 for validation. There is no specific reason in the number. I just tried to use as much image as possible for training the model hoping it could be robust enough, while also keeping a sufficient number of testing images in order to validate the model. I found that this ratio could reach a reasonable result hence I kept using this ratio.**
        + Create a convolutional neural network and train it using the dataset created in step 2. The model should not have more than two million parameters. You don't need to train the model for a long time, just a few iterations and to allow the loss to converge.
        + How did you calculate the number of parameters?
            + **I calculated the parameters by summing the number of elements in every parameter group of a model using: ```sum(p.numel() for p in model.parameters()```. The model consists of 1,049,073 parameters.**
        + Which loss function did you use?
            + **MSELoss function.**
        + How many epochs was the model trained?
            + **10 epochs, but it started to overfit after 6th epoch.**
        + What was the loss before and after training?
            + **This link: https://github.com/iluntsai99/Correlation-Prediction/blob/main/log shows the log to my code. Without any training, the model initiated with loss 0.001885. After training for 6 epochs, the validation loss decreased to 0.000008.**
