
# SiamGauss
This code base provides the tools to create, test, and use SiamGauss discussed in SiamGauss: A Siamese Surrogate Model for Predicting Dominance in Multi-Objective Optimisation. The test cases include multi-objective benchamark problems from and https://arxiv.org/abs/2402.02033 and NATS bench Neural Network architecture search. Test cases include:

**NATS**
1. **SNN creation and full training on NATS dataset**
2. **SNN fine-tuning**
3. **Testing of SNN performance**
**MOMOP**
1. **Training and testing on all 11 MOMOP Problems presented in the CEC competition**

### Requirements

- Python 3.11.6 (Likely functional on < Python 3.9, however, currently untested)

- Install the required packages:
```bash
   pip install -r requirements.txt
```


# NATS

## Prerequisites

(For NATS use): Before beginning, please ensure you download the appropriate datasets from [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects) . Documentation is contained there. These datasets must be extracted and the file location updated in paths.py


1. **update NATS dataset files paths in paths **
2. **Note which .pt SNN archititure you plan on using, this must be updated in the 'main' function of your chosen action**

## SNN creation and full training on NATS dataset(Siamese_for_dominance_check.py):
The following steps will guide you to make your own SNN from scratch and train it. NOTE: THIS IS RESOURCE INTENSIVE, AND CARE IS RECOMMEND RUNNING THIS SCRIPT OUTSIDE OF A CLUSTER

Set-up:
1. **ensure correct database is attached, if you have renamed the dataset, modify paths.py appropriatley.**
2. **setting up hyper parameters, all parameters can be found in the argument parser, the 3 most with the largest impact however are:**
```python
    args.lr = 0.0001
    args.epochs = 10
    intial_dataset_size = 10
```
**These settings are low weight to reduce risk of failure if run with without modification, in practice larger datasets will result in better results if compuational power is available**
3. **Update**
```python 
model_name = 'siamese_network.pt' 
``` 
**in paths.py function as required**

4. **loading and saving the model, in train.py the arg paser load_model = True, save_model=True must be modified as required. If both are true the model will be over written, so ensure models are copied if this is not desired.**

Then to run the code simply use the command, ensure you are in the root directory and then:
```bash
python -m SiamGauss.NATS.train
```
in  the terminal.

## SNN fine-tuning (fine_tuning.py):
The hyper parameters and set of the fine tuning script match that of the full training. The only modification is that the NN archiecutre IDs are passed in giving more control over what dataset is used for fine-tuning. 

The intention for this is to be used a function within a search script, however for basic use a simple main function is provided. To use simply modify 
```python
data_points_idx = [i for i in range(0,10)] 
```
to contain the list of desired NATS architeture IDs, and run with (ensure you are in the root directory):
```bash
python -m SiamGauss.NATS.fine_tuning
```

## Testing of SNN performance (Testing.py)
Testing of the performace provides the loss, accuracy and F1-score of the SNN. 

Set-up
1. To test the network, first update num_tests=1000 to the appropraite number of test you wish to run (note: if this is larger than the dataset an out of bounds error will occur.)
2. Modify the model name in the main function to load in the SNN you wish test.
3. Then run `python -m Testing`

## CEC Benchmarks
To modify SNN arch, update accordingly:
```python
model = SiameseNetwork_dominance(input_size = problem.D,num_repeated_hidden = 1, hidden_size_1 = problem.D*8*8, hidden_size_2 =  problem.D, fc_size =  4, convD = 0).to(device)
```
For running SiamGauss on the first (most simple) MOMOP problem,`python -m SiamGauss.Benchamrks.train`
For running SiamGauss on all MOMOP problems,`python -m SiamGauss.Benchamrks.train_all`
