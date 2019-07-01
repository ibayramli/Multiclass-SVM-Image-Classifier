# SVM-Image-Classifier

In this repo, I am building an linear image classifier using Multiclass Support Vector Machine. Training a SVM consists of finding an optimal matrix W that given 3072-dimensional image and bias vectors outputs a 10-dimensional class scores vector. In that vector, we aim to have a prominent margin between the score of correct class and the incorrect ones. To this end, I set up a vectorized loss function that punishes the classifier for the images where the magnitude of the margin between correct and incorrect classes falls below our predefined constant Delta. Then, I implement Stochastic Gradient Descent using the analytic gradient of the loss function to find the matrix minimizing this loss. Lastly, I perform cross-validation to tune the regularization and learning rate parameters. The final classifier, when trained with 49,000 examples, achieves 36% accuracy. 

To run this code, you must first obtain the CIFAR-10 dataset by first running the following shell code:

wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

tar -xzvf cifar-10-python.tar.gz

rm cifar-10-python.tar.gz

The structure of the documents and data importing tools (data_utils.py) are borrowed from Stanford's CS231n: *Convolutional Neural Networks* class assignments. 
