# Crystallization Mechanisms From Machine Learning
This repository hosts a collection of custom code and scripts accompanying the journal publication:

["Uncovering the effects of interface-induced ordering of liquid on crystal growth using Machine Learning"  
Rodrigo Freitas and Evan Reed  
Nature Communications  
arXiv:1909.05915](https://arxiv.org/abs/1909.05915).

If the script you need is not included here please contact the authors. Any code related to the paper will either be added to this repository (if it is of sufficient general interest) or sent directly to you.

## Requirements
In order to run the scripts contained here you will need to install [Ovito](https://www.ovito.org). If you only install Ovito's module through pip make sure to also have a compatible version of [numpy](https://numpy.org), [scipy](https://scipy.org), and [scikit-learn](https://scikit-learn.org) also installed. Otherwise, if you downloaded Ovito's Python interpreter (named "ovitos") make sure to install scipy and scikit-learn modules:
```
ovitos -m pip install scipy sklearn
```

## Usage
Use ovitos (or Python if you only installed the Ovito module) to run the *compute.py* script.
```
ovitos compute.py
```
This script will load the Molecular Dynamics snapshot in *input_data* and compute the value of the alpha parameter and softness of each atom. See paper for details on the definition of these quantities. Notice that it might take a few minutes to run the script as the simulation snapshot contains 500,000 atoms.

After *compute.py* is run it will output a Molecular Dynamics snapshot in *output_data* where each particle will have an alpha and softness property value. In order to visualize the result open the *colored_by_softness.ovito* file using Ovito's GUI "Load Program State". The visualization should be very similar to Fig.3c of the paper, the only difference is that in the paper we used the time averaged alpha in order to identify crystal atoms, while here we use the instantaneous alpha since only one snapshot is available.

## Authors & Contact

Rodrigo Freitas | freitas@stanford.edu  
Evan Reed | evanreed@stanford.edu
