import numpy as np
# ---------------------attack config------------------------#

# epsilon = lambda: np.abs(np.random.normal(0,0.25,1))[0]
epsilon = lambda: 0.1
eps_att = 0.1
eps_def = 0.3
#
attack_params = {
    "FGSM_MNIST": {
    'epsilon' : eps_att,
    'order': np.inf,
    'clip_max': None,
    'clip_min': None
    },

    "FGSM_CIFAR10": {
    'epsilon' : eps_att,
    'order': np.inf,
    'clip_max': None,
    'clip_min': None
    },

    "FGSM_CIFAR100": {
    'epsilon' : eps_att,
    'order': np.inf,
    'clip_max': None,
    'clip_min': None
    },

    "FGSM_tiny_imagenet": {
    'epsilon' : eps_att,
    'order': np.inf,
    'clip_max': None,
    'clip_min': None
    },

    "PGD_CIFAR10": {
    'epsilon': eps_att,
    'clip_max': 1.0,
    'clip_min': 0.0,
    'print_process': True
    },

    "PGD_CIFAR100": {
    'epsilon' : eps_att,
    'clip_max': 1.0,
    'clip_min': 0.0,
    'print_process': False
    },

    "PGD_tiny_imagenet": {
    'epsilon': eps_att,
    'clip_max': 1.0,
    'clip_min': 0.0,
    'print_process': True
    },

    "LBFGS_MNIST": {
    'epsilon': eps_att,
    'maxiter': 20,
    'clip_max': 1,
    'clip_min': 0,
    'class_num': 10
    },

    "CW_MNIST": {
    'confidence': 1e-4,
    'clip_max': 1,
    'clip_min': 0,
    'max_iterations': 1000,
    'initial_const': 1e-2,
    'binary_search_steps': 5,
    'learning_rate': 5e-3,
    'abort_early': True,
    },

    "DeepFool_CIFAR10": {

    }

}

#-----------defense(Adversarial training) config------------#

defense_params = {
    "PGD_MNIST":{
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name' : "mnist_pgdtraining_0.3.pt",
        'epsilon' : eps_def,
        'epoch_num' : 150,
        'lr' : 0.1
    },

    "PGD_CIFAR10":{
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name' : "mnist_pgdtraining_0.3.pt",
        'epsilon' : eps_def,
        'epoch_num' : 150,
        'lr' : 0.1
    },

    "PGD_CIFAR100":{
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name' : "mnist_pgdtraining_0.3.pt",
        'epsilon' : eps_def,
        'epoch_num' : 150,
        'lr' : 0.1
    },

    "PGD_tiny_imagenet":{
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name' : "mnist_pgdtraining_0.3.pt",
        'epsilon' : eps_def,
        'epoch_num' : 150,
        'lr' : 0.1
    },

    "FGSM_MNIST":{
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name' : "mnist_fgsmtraining_0.2.pt",
        'epsilon' : eps_def,
        'epoch_num' : 150,
        'lr_train' : 0.1
    },

    "FGSM_CIFAR10":{
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name' : "mnist_fgsmtraining_0.2.pt",
        'epsilon' : eps_def,
        'epoch_num' : 150,
        'lr_train' : 0.1
    },
    "FGSM_CIFAR100":{
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name' : "mnist_fgsmtraining_0.2.pt",
        'epsilon' : eps_def,
        'epoch_num' : 150,
        'lr_train' : 0.1
    },

    "FGSM_tiny_imagenet":{
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name' : "mnist_fgsmtraining_0.2.pt",
        'epsilon' : eps_def,
        'epoch_num' : 150,
        'lr_train' : 0.1
    },

    "FAST_MNIST":{
        'save_dir': "./defense_model",
        'save_model': True,
        'save_name' : "fast_mnist_0.3.pt",
        'epsilon' : eps_def,
        'epoch_num' : 150,
        'lr_train' : 0.1
    }
}

