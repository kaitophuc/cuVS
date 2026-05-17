from config import DISTRIBUTION_MODE, K, METRIC, N_LISTS, N_PROBES_SWEEP

def describe_ivf_params(dataset, print_info=True):
    num_vectors = dataset.shape[0]
    dim = dataset.shape[1]

    average_list_size = num_vectors / N_LISTS

    if print_info:
        print("IVF-Flat parameter setup")
        print("Number of database vectors:", num_vectors)
        print("Vector dimension:", dim)
        print("Metric:", METRIC)
        print("k:", K)

        print("n_lists:", N_LISTS)
        print("Average vectors per IVF list:", average_list_size)

        print("n_probes sweep:", N_PROBES_SWEEP)
        print("Distribution mode:", DISTRIBUTION_MODE)

        print()
        print("Meaning:")
        print("n_lists controls how many coarse clusters/buckets the dataset is split into.")
        print("n_probes controls how many of those buckets are searched for each query.")
        print("Higher n_probes usually gives better recall but slower search.")
        print("Lower n_probes usually gives faster search but lower recall.")

    return num_vectors, dim, average_list_size

def main():
    from load_data import load_default_data

    _, dataset, _, _ = load_default_data(print_info=True)
    describe_ivf_params(dataset, print_info=True)

if __name__ == "__main__":
    main()
