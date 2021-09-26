# MinMax-Sampling

## Federated Learning

Codes in this folder are forked from open source code of *FetchSGD: Communication-Efficient Federated Learning with Sketching*. We apply our Min-Max Sampling in the aggregation process to evaluate performance.

In cifar10.sh we provided an example of how to run the program. The meaning of some of the arguments are provided in the chart below: 

| Argument name      | meaning                                                               |
|--------------------|-----------------------------------------------------------------------|
| dataset_dir        | directory to where you store the dataset                              |
| num_epochs         | total number of epochs                                                |
| num_clients        | total number of clients                                               |
| num_workers        | the number of clients participate each turn                           |
| num_devices        | number of gpus                                                        |
| num_rows, num_cols | in FetchSGD, it is used to set the sketch size to num_rows * num_cols |
| k                  | sample size in local topk and our algorithm                           |

For FEMNIST, because that we have the orignal data preprocessed into 3597 clients, so the num_clients is always equal to 3597. We run for only 1 epoch for FEMNIST.
