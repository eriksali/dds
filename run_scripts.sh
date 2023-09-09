#!/bin/bash
python cifar10/certify.py --sigma 0.12 --skip 400 --N0 100 --N 10000 --batch_size 10 --output results/cifar10_sigma_0.12
python cifar10/certify.py --sigma 0.25 --skip 400 --N0 100 --N 10000 --batch_size 10 --output results/cifar10_sigma_0.25
python cifar10/certify.py --sigma 0.50 --skip 400 --N0 100 --N 10000 --batch_size 10 --output results/cifar10_sigma_0.50
python cifar10/certify.py --sigma 1.00 --skip 400 --N0 100 --N 10000 --batch_size 10 --output results/cifar10_sigma_1.00
python cifar10/analyze.py