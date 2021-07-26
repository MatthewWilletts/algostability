#### MLP RUNS

# run VaDE - MLP MNIST
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation -z 200
# run iVAE - MLP MNIST
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 200
# calculate VaDE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation -z 200 --all --mcc
# calculate iVAE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 200 --all --mcc
# plot MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation -z 50 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation -z 90 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_mlp.yaml --n-sims 10 --representation -z 200 --all --mcc --plot

# run VaDE - MLP CIFAR10
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation -z 200
# run iVAE - MLP CIFAR10
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 200
# calculate VaDE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation -z 200 --all --mcc
# calculate iVAE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 200 --all --mcc
# plot MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation -z 50 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation -z 90 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_mlp.yaml --n-sims 10 --representation -z 200 --all --mcc --plot

# run VaDE - MLP SVHN
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation -z 200
# run iVAE - MLP SVHN
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 200
# calculate VaDE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation -z 200 --all --mcc
# calculate iVAE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation --baseline --ivae -z 200 --all --mcc
# plot MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation -z 50 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation -z 90 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_mlp.yaml --n-sims 10 --representation -z 200 --all --mcc --plot



#### CONV RUNS

# run VaDE - CONV MNIST
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation -z 200
# run iVAE - CONV MNIST
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 200
# calculate VaDE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation -z 200 --all --mcc
# calculate iVAE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 200 --all --mcc
# plot MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation -z 50 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation -z 90 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config mnist_convmlp.yaml --n-sims 10 --representation -z 200 --all --mcc --plot

# run VaDE - CONV CIFAR10
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation -z 200
# run iVAE - CONV CIFAR10
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 200
# calculate VaDE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation -z 200 --all --mcc
# calculate iVAE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 200 --all --mcc
# plot MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation -z 50 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation -z 90 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_convmlp.yaml --n-sims 10 --representation -z 200 --all --mcc --plot

# run VaDE - CONV SVHN
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation -z 200
# run iVAE - CONV SVHN
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 200
# calculate VaDE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation -z 200 --all --mcc
# calculate iVAE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation --baseline --ivae -z 200 --all --mcc
# plot MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation -z 50 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation -z 90 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_convmlp.yaml --n-sims 10 --representation -z 200 --all --mcc --plot



#### RESNET RUNS

# run VaDE - RESNET CIFAR10
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation -z 200
# run iVAE - RESNET CIFAR10
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 200
# calculate VaDE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation -z 200 --all --mcc
# calculate iVAE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 200 --all --mcc
# plot MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation -z 50 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation -z 90 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config cifar10_resnet.yaml --n-sims 10 --representation -z 200 --all --mcc --plot

# run VaDE - RESNET SVHN
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation -z 200
# run iVAE - RESNET SVHN
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 50
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 90
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 200
# calculate VaDE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation -z 200 --all --mcc
# calculate iVAE MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 50 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 90 --all --mcc
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation --baseline --ivae -z 200 --all --mcc
# plot MCCs
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation -z 50 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation -z 90 --all --mcc --plot
CUDA_VISIBLE_DEVICES=0 python -u main.py --config svhn_resnet.yaml --n-sims 10 --representation -z 200 --all --mcc --plot

