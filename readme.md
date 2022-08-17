# Contents

* [Acknowledgement](#acknowledgement)
* [Citing This Paper](#citing-this-paper)
* [Abstract](#abstract)
* [Environment Setup](#environment-setup)
* [Running Framework](#running-framework) 
* [Model Zoo](#model-zoo)


## Acknowledgement 
This repository contains the source code for the `Pruning Adversarially Robust Neural Networks without Adversarial Examples` project developed by the Northeastern University's SPIRAL research group. This research was generously supported by the National Science Foundation (grant CCF-1937500). 


## Citing This Paper
Please cite the following paper if you intend to use this code for your research.
> T. Jian*, Z. Wang*, Y. Wang, J. Dy, S. Ioannidis, "Pruning Adversarially Robust Neural Networks without Adversarial Examples", TBD, 2022.

## Abstract
Adversarial pruning compresses models while preserving robustness. Current methods require access to adversarial examples during pruning. This significantly hampers training efficiency. Moreover, as new adversarial attacks and training methods develop at a rapid rate, adversarial pruning methods need to be modified accordingly to keep up. In this work, we propose a novel framework to prune a previously trained robust neural network while maintaining adversarial robustness, without further generating adversarial examples. We leverage concurrent self-distillation and pruning to preserve knowledge in the original model as well as regularizing the pruned model via the Hilbert-Schmidt Information Bottleneck. We comprehensively evaluate our proposed framework and show its superior performance in terms of both adversarial robustness and efficiency when pruning architectures trained on the MNIST, CIFAR-10, and CIFAR-100 datasets against five state-of-the-art attacks.

## Environment Setup
Please install the python dependencies and packages found below:
```bash
pytorch-1.6.0
torchvision-0.7.0
numpy-1.16.1
scipy-1.3.1
tqdm-4.33.0
yaml-0.1.7
torchattacks
```

Please setup environment in the project root directory using:
```bash
source env.sh
```

After intalling "torchattacks" package, we need to modify one place as follows to make sure our framework work. Please go to the installed package directory (`/.../torchattacks/attacks/`), modify `pgd.py` by finding the line `outputs = self.model(adv_images)`, and insert the following code after it:
```bash
if type(outputs) == tuple:
    outputs = outputs[0]
```

## Running Framework
You could produce the main results of Table 2, 3, 4, 5 & Figure 3 (PwoA) by this repository. We will release the full version of the repository soon. 

To setup adversarially robust pre-trained models for pruning, we consider five adversarially trained models provided by open-source state-of-the-art work, summarized in Table 1 in our paper. Please download those models from their original repository and saved in './assets/models/' under this repo.

To reproduce experiments that we have in the paper, one could run our batch script by the following instruction:
```bash
mnist.sh         # PwoA on MNIST
cifar10.sh       # PwoA on CIFAR-10
cifar100.sh      # PwoA on CIFAR-100
```

Please refer to [./bin/run_hbar](./run_hbar) for more usages. The arguments in the code are self-explanatory.
