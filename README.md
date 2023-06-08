# Project: Evaluating the Performance of the Adam Optimizer:

This project is based on the reproduction of the experiments conducted in the papers "Adam: A Method for Stochastic Optimization" (arXiv:1412.6980) and "Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification" (arXiv:2012.11212).

## Dependencies
### Requirements

The code uses the PyTorch library, among others. You will need to have the following libraries installed in your Python environment:

```python
torch
torchvision
matplotlib
```

You can install the above libraries using pip:
```bash
pip install torch torchvision 
pip install matplotlib

```

### Files

For the code to run successfully, the following model files should be present:

- improved_pretrained_model.pth # use any pr-etrained model and make sure to give it the name as specified here, we used Resnet50 model.
- improved_model.pth # you can use the provided custom model in the files section.

Please make sure to update the file paths as it is stored in your device.

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
3. After making sure that all libraries are installed, task one can be executed with 0 errors.

#### The results of task2 expected are as follows:

###### 1) model1: Logistic Regression with weight_decay=0.0001, beta1=0.9, beta2=0.999, epsilon=0.0001
    - Epoch 10/10, Train Loss: 0.2511, Train Acc: 93.09%, Test Loss: 0.2611, Test Acc: 92.63%
    - 9263 correct and 737 incorrect predictions.
- <img width="680" alt="Screen Shot 2023-06-04 at 16 36 38" src="https://github.com/omniaelmenshawy/StochasticOptimizationUsingAdam/assets/77496383/33b8916f-af03-4435-924b-72279ed024ad">

- <img width="679" alt="Screen Shot 2023-06-04 at 16 39 23" src="https://github.com/omniaelmenshawy/StochasticOptimizationUsingAdam/assets/77496383/62f04ea4-c2e5-47c1-9dc1-33c83cf20b60">


###### 2) model2: Multi Layer Network with weight_decay=0.0001, beta1=0.9, beta2=0.999, epsilon=0.0001
    - Epoch 10/10, Train Loss: 0.0656, Train Acc: 97.87%, Test Loss: 0.0644, Test Acc: 98.13%
    - 9813 correct and 187 incorrect predictions.
- <img width="680" alt="Screen Shot 2023-06-04 at 16 37 47" src="https://github.com/omniaelmenshawy/StochasticOptimizationUsingAdam/assets/77496383/33b41c24-b3e3-47d0-bfba-c2d866c5181a">
- <img width="679" alt="Screen Shot 2023-06-04 at 16 39 38" src="https://github.com/omniaelmenshawy/StochasticOptimizationUsingAdam/assets/77496383/23d27002-157b-4fe8-85e6-524aae1d6caa">


###### 3) model3: ConvNet with weight_decay=0.0001, beta1=0.9, beta2=0.999, epsilon=0.0001
      - 9919 correct and 81 incorrect predictions.
      - Epoch 10/10, Train Loss: 0.0034, Train Acc: 99.89%, Test Loss: 0.0313, Test Acc: 99.16%


<img width="679" alt="Screen Shot 2023-06-04 at 16 38 48" src="https://github.com/omniaelmenshawy/StochasticOptimizationUsingAdam/assets/77496383/79596251-0efd-45c1-a2b2-5732ba105fef">

<img width="679" alt="Screen Shot 2023-06-04 at 16 40 06" src="https://github.com/omniaelmenshawy/StochasticOptimizationUsingAdam/assets/77496383/36efe691-ce37-4313-81a0-9920d28aee0a">

##### 4) Task1 Overall result: 

<img width="679" alt="Screen Shot 2023-06-04 at 16 40 32" src="https://github.com/omniaelmenshawy/StochasticOptimizationUsingAdam/assets/77496383/d469b41c-b68f-40b2-89b1-4c8ce4d43474">


### Task 2: Implementing a Deep Feature Space Trojan (DFST) Attack that is based on Adam optimization:

This task is inspired by the attack strategies in the "Deep Feature Space Trojan Attack of Neural Networks by Controlled Detoxification" paper.

1. In the second task, the Adam optimizer is used to implement a DFST attack on two CNN models trained on the CIFAR10 dataset.

2. Use the two trained models (`improved_pretrained_model.pth` and `improved_model.pth`). 

3. Implement the attack using a CycleGan architecture to generate a trigger. The network uses Adam as its main optimizer for both the generator and discriminator. 

4. Use a 10% poisoning ratio to poison the dataset with the generated trigger. 

5. Finally, evaluate the attack by running your evaluation code. 

#### The results of task2 expected are as follows:

#### Model1 Accuracy before Attack: 73.96%
#### Model1 Accuracy under attack: 62.34%
#### Model2 Accuracy before Attack: 84.62%
#### Model2 Accuracy under attack: 69.53%  

<img width="679" alt="Screen Shot 2023-06-04 at 16 41 01" src="https://github.com/omniaelmenshawy/StochasticOptimizationUsingAdam/assets/77496383/b399f670-08c7-4e83-9dc3-cd557c55577b">

<img width="679" alt="Screen Shot 2023-06-04 at 16 41 21" src="https://github.com/omniaelmenshawy/StochasticOptimizationUsingAdam/assets/77496383/67823938-8194-4829-9856-078a0cf8ee7e">

Please note that you may not get exactly the same results due to randomness in training processes. This is a common situation in machine learning experiments.

This project serves as a practical implementation and evaluation of the concepts presented in both papers. It is a step towards understanding the capabilities and performance of the Adam optimizer, as well as its potential applications in different aspects of machine learning, including adversarial attacks.
