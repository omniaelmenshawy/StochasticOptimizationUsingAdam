# Project: Evaluating the Performance of the Adam Optimizer and Implementing a DFST Attack

This project is based on the reproduction of the experiments conducted in the papers "Adam: A Method for Stochastic Optimization" (arXiv:1412.6980) and "Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification" (arXiv:2012.11212).

## Requirements

The code uses the PyTorch library, among others. You will need to have the following libraries installed in your Python environment:

```python
torch
torchvision
matplotlib
```

You can install the above libraries using pip:
```bash
pip install torch torchvision matplotlib
```

## Files

For the code to run successfully, the following model files should be present:

- improved_pretrained_model.pth
- improved_model.pth

## Steps to Run the Project

### Task 1: Adam optimizer tests

This task is inspired by the experimental evaluations in the "Adam: A Method for Stochastic Optimization" paper.

1. The first task involves testing the Adam optimizer on various models: a Logistic Regression model, a Multi-layer neural network, and a Convolutional Neural Network model.

2. Next, a sensitivity analysis is conducted on the models, using different hyperparameters for the Adam optimizer. You can set the values in your code as follows:

```python
weight_decays = [1e-4, 1e-3]
beta1_values = [0.9, 0.99]
beta2_values = [0.999, 0.9999]
epsilons = [1e-6, 1e-4]
```

### Task 2: Implementing a Deep Feature Space Trojan (DFST) Attack

This task is inspired by the attack strategies in the "Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification" paper.

1. In the second task, the Adam optimizer is used to implement a DFST attack on two CNN models trained on the CIFAR10 dataset.

2. Use the two trained models (`improved_pretrained_model.pth` and `improved_model.pth`). 

3. Implement the attack using a CycleGan architecture to generate a trigger. The network uses Adam as its main optimizer for both the generator and discriminator. 

4. Use a 10% poisoning ratio to poison the dataset with the generated trigger. 

5. Finally, evaluate the attack by running your evaluation code. 

The results expected are as follows:

#### Model1 Accuracy before Attack: 73.96%
#### Model1 Accuracy under attack: 62.34%
#### Model2 Accuracy before Attack: 84.62%
#### Model2 Accuracy under attack: 69.53%  

Please note that you may not get exactly the same results due to randomness in training processes. This is a common situation in machine learning experiments.

This project serves as a practical implementation and evaluation of the concepts presented in both papers. It is a step towards understanding the capabilities and performance of the Adam optimizer, as well as its potential applications in different aspects of machine learning, including adversarial attacks.
