# echo 'start 10000'


python3 cv_train.py --dataset_dir ~/cifar10 --local_batch_size 5  --dataset_name CIFAR10 --model ResNet9   --local_momentum 0.0  --virtual_momentum 0.9 --weight_decay 1e-4 --mode sketch --num_clients 2000 --num_devices 1 --num_workers 100 --k 100000 --num_rows 1  --num_cols 600000 --share_ps_gpu --num_epochs 48 --outlier_thres 100 --error_type virtual --pivot_epoch 10
# python3 cv_train.py --dataset_dir ~/cifar10 --local_batch_size 25  --dataset_name CIFAR10 --model ResNet9   --local_momentum 0.0  --virtual_momentum 0.9 --weight_decay 1e-4 --mode sampling --iid --num_clients 1000 --num_devices 1 --num_workers 100 --k 1000000  --num_rows 1  --num_cols 1000000 --share_ps_gpu --num_epochs 10 --topk_down
#python3 cv_train.py --dataset_dir ~/cifar10 --local_batch_size 25   --dataset_name CIFAR10 --model ResNet9   --local_momentum 0.0  --virtual_momentum 0.9 --weight_decay 1e-4 --mode local_topk --iid --num_clients 1000 --num_devices 1 --num_workers 100 --k 100000  --num_rows 1  --num_cols 300000 --share_ps_gpu --num_epochs 100 --error_type local

# python3 cv_train.py --dataset_dir ~/cifar10 --local_batch_size 25   --dataset_name CIFAR10 --model ResNet9   --local_momentum 0.0  --virtual_momentum 0.9 --weight_decay 1e-4 --mode sketch --iid --num_clients 1000 --num_devices 1 --num_workers 100 --k 10000  --num_rows 1  --num_cols 300000 --share_ps_gpu --num_epochs 100 --error_type virtual
