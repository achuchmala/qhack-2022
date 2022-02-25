# qhack-2022

Trial of implementation of quantum-classical Wasserstein GAN (WGAN) for detection of anomalies in cred card transactions based on the paper:

https://arxiv.org/abs/2010.10492

Unfortunately due to lack of time, only some intial steps were completed:
1. The implementation of Wasserstein GAN with initial layer in the form of variational
 quantum circuit.
2. Preparation of WGAN training procedure.

The presented implementation lacks the following:
1. Training of WGAN is not completed.
2. Determination of Anomaly Score for AnoGAN, used in anomalies detection optimization, through optimization of latent variables.

Despite the fact that it this code is in very early stage and needs thorough code refinement and addition of lacking features, it will be foundation for tutorial in the topic of using Pennylane with Pytorch for quantum-classical neural networks. Work is still in progress.

NOTE: The file with data must be in the same folder where the 'quantum-wgan.py' script is located. Used data are available at https://www.kaggle.com/mlg-ulb/creditcardfraud. In order to not upload large files to the repo they were not included.

Resources:
https://arxiv.org/abs/2010.10492

https://pennylane.ai/qml/demos/tutorial_quantum_gans.html

https://arxiv.org/abs/1406.2661
