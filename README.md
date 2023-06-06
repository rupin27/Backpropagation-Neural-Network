# Backpropagation-Neural-Network
This repository contains the implementation of the backpropagation algorithm to train a neural network from scratch. The implementation supports networks with an adjustable number of layers and neurons. It also includes a regularization mechanism to mitigate overfitting. The performance of the neural network is evaluated using the stratified cross-validation strategy with k = 10 folds. The project includes two datasets for evaluation: the Wine Dataset and the 1984 United States Congressional Voting Dataset.

# Requirements
<p>
  <i>Python 3.x</i><br>
  <i>NumPy</i><br>
  <i>Pandas</i>
</p>

## Usage

### 1. Installation
Clone the repository to your local machine:
> git clone https://github.com/rupin27/Backpropogation-Nueral-Network.git
### 2. Dataset Preparation
Place the dataset files in the `datasets` directory. Ensure that the datasets are properly formatted and contain the necessary attributes and labels.

### 3. Running the Backpropagation Algorithm
To run the backpropagation algorithm and evaluate the neural network on the datasets, follow these steps:
1. Set the desired network architecture, number of layers, number of neurons, and regularization parameter in the `results.py` file.

2. Open a terminal or command prompt and navigate to the project directory.

3. Run the following command to execute the backpropagation algorithm:
python3 backprop_example.py
4. The algorithm will train the neural network using the specified configuration and display the progress and performance metrics during training.

5. Once the training is complete, the algorithm will output the average performance of the neural network on the datasets.

### 4. Correctness Verification
To verify the correctness of the backpropagation implementation, the provided benchmark files (`backprop_example1.txt` and `backprop_example2.txt`) can be used. The implementation should produce similar intermediate quantities and outputs as described in these files.

1. Open the `backprop_example.py` file.

2. Ensure that the file paths for the benchmark files (`backprop_example1.txt` and `backprop_example2.txt`) are correctly specified.

3. Run the following command to execute the correctness verification:
> python3 backprop_example.py
4. The verification function will compare the intermediate quantities computed by the backpropagation algorithm with those provided in the benchmark files. If the implementation is correct, the function will print the relevant quantities for each training instance.

## Results

The results of the backpropagation algorithm will be displayed during training and summarized at the end. The performance of the neural network on the Wine Dataset and the 1984 United States Congressional Voting Dataset will be reported. The impact of different design decisions, such as the number of layers, number of neurons, and regularization parameter, on the performance will be analyzed. To view my in depth analysis of the model on the dataset, review the report here. [Click here](/Report.pdf)
