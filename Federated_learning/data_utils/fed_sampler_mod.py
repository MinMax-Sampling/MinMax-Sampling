import numpy as np

__all__ = ["FedSampler"]


class FedSampler:
    """ Samples from a federated dataset

    Shuffles the data order within each client, and then for every
    batch requested, samples num_workers clients, and returns
    local_batch_size data from each client.
    """

    def __init__(self, dataset, num_workers, local_batch_size,
                 shuffle_clients=True):
        self.dataset = dataset
        self.num_workers = num_workers
        self.local_batch_size = local_batch_size
        self.shuffle_clients = shuffle_clients

    def __iter__(self):
        data_per_client = self.dataset.data_per_client
        cumsum = np.cumsum(data_per_client)
        cumsum = np.hstack([[0], cumsum])
        # permute the data indices within each client

        type_sum = [5000 * i for i in range(10)]
        permuted_data = []
        for i in range(len(data_per_client)):
            perm = []
            u = data_per_client[i]
            v = i // (len(data_per_client) // 10)
            major_num = u // 25 * 16
            minor_num = u // 25 * 1
            for k in range(10):
                if (k == v):
                    perm.extend(
                        np.arange(type_sum[k], type_sum[k] + major_num))
                    type_sum[k] += major_num
                else:
                    perm.extend(
                        np.arange(type_sum[k], type_sum[k] + minor_num))
                    type_sum[k] += minor_num
            # print(len(perm))
            permuted_data.extend(perm)

        # need to keep track of where we are within each client
        cur_idx_within_client = np.zeros(self.dataset.num_clients,
                                         dtype=int)

        def sampler():
            while True:

                u = np.sum(cur_idx_within_client)
                v = np.sum(data_per_client)
                # print(data_per_client, cur_idx_within_client)
                # only choose clients that have any data left
                nonexhausted_clients = np.where(
                    cur_idx_within_client < data_per_client
                )[0]
                if len(nonexhausted_clients) == 0:
                    break
                num_workers = min(self.num_workers,
                                  len(nonexhausted_clients))
                # choose randomly from the clients that have data left
                print(num_workers)
                workers = np.random.choice(nonexhausted_clients,
                                           num_workers,
                                           replace=False)
                # figure out how much data each chosen client has left
                records_remaining = (data_per_client[workers]
                                     - cur_idx_within_client[workers])

                # choose up to local_batch_size elements from each worker
                if self.local_batch_size == -1:
                    # local batch size of -1 indicates we should use
                    # the client's entire dataset as a batch
                    actual_batch_sizes = records_remaining
                else:
                    actual_batch_sizes = np.clip(records_remaining,
                                                 0,
                                                 self.local_batch_size)
                r = np.hstack([
                    permuted_data[s:s + actual_batch_sizes[i]]
                    for i, s in enumerate(cumsum[workers] +
                                          cur_idx_within_client[workers])
                ])
                if self.local_batch_size != -1:
                    assert r.size <= self.num_workers * self.local_batch_size
                yield r
                cur_idx_within_client[workers] += actual_batch_sizes

        return sampler()

    def __len__(self):
        return len(self.dataset)
