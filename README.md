# cnn

The project is an implementation of a convolutional neural network for handwritten digit classification. 

The project is implemented in C++ programming language. It only has a source file named convnet.cpp. 

When running the application, you may have to wait some time until the network is trained. After the training is done, the application will output the results of the network when different inputs (from either train or test data) are applied and the correct label. 

The train-images.idx3-ubyte and train-labels.idx1-ubyte files must be in the same directory as the executable file. 

The application can work in many modes, by modifying the value of different variables in the source file (will be detailed below). 


The first few lines in the source file contains some constants that can be modified to alter the behaviour of the network. 


The goal of the convolutional neural network is to learn to correctly classify the handwritten digits in images; more precisely, given an image, it should identify the digit (from 0 to 9) written it.  
The convolutional neural network uses the MNIST dataset, which is a database of 20000 images of handwritten digits, and the correct digit in each image. This data is divided into training data and test data: for speed up purposes, only NR_EXAMPLES_READ examples can be read from the database; a fraction of RATIO_TRAIN_TEST of the read examples is used for training, and the remaining ones are only used for test. 
In the first stage, the network is trained using the training data; for a number of NR_EPOCHS iterations (epochs), each example is applied as input to the network, and the backpropagation algorithm is used to train the network using on the given correct answer (label) from the database. 
In the test stage, each example is applied to the network and the output is observed and compared to the correct/desired output. First we test the network against the training examples, and then against the test examples, which the network "has never seen" before. 
The application functions in many different modes, given by the variable MODE, which can take the following values: 
	TRAIN_AND_TEST 	- the network is trained, then tested
	TRAIN 			- the network is only trained, and the final trained state of the network is written to a file
	TEST 			- the network is initialised from a file (presumably a previously trained network) and then tested
	TRAIN_CONTINUE 	- the network is read from a file and the training process continues from that state on for yet another NR_EPOCHS epochs
The names of the files to read from or write to the network are given by the variables FILENAME_IN and FILENAME_OUT, respectively. 
You should not modify the value of RATIO_TRAIN_TEST or NR_EXAMPLES_READ between a training phase and a test phase. 

