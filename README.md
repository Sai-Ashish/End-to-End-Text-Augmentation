# End-to-End-Text-Augmentation
TACL paper- "A Multi-Level Optimization Framework for End-to-End Text Augmentation" code

This repository is the code for end-to-end data augmentation. 

The components of the code are as follows,

1) arhitect_adam.py contains the code for the optimization. 
2) attention_params.py is for the attention parameters.
3) BART.py contains the conditional text generation BART model for data augmentaiton.
4) ClassifierModel.py is our text classification model. 
5) data_set.py is the file to load the related datasets. 
6) utils.py contains the necessary utilities. 

We have to run arch_search_adam.py for training the end-to-end model. The code given is the general framework code and can be replaced with other models/datasets. We can also finetune the parameters according to the downstream task/dataset and models.
