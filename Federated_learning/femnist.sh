python3.7 cv_train.py\
	--mode uncompressed \
	--num_clients 3597 \
	--num_workers 20 \
	--num_epochs 1 \
	--pivot_epoch 0.2 \
	--local_batch_size 30 --dataset_dir ~/leaf-master/data/femnist/data --dataset_name EMNIST --model ResNet101LN  --local_momentum 0.0  --virtual_momentum 0.9 --weight_decay 1e-4 --num_devices 1 --num_rows 1 --share_ps_gpu --outlier_thres 100 --valid_batch_size 512
