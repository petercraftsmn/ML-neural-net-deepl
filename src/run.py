from src.pk_network import Network


def main():
    sizes = [2, 3, 1]
    network = Network(sizes)
    print("num_layers =", network.num_layers)
    print("sizes =", sizes)
    print("sizes[1:] =", sizes[1:])
    print("sizes[:-1] =", sizes[:-1])
    print("biases =", network.biases)
    print("biases = [np.random.randn(y, 1) for y in sizes[1:]] ->", [(y, 1) for y in sizes[1:]])
    print("weight =", network.weights)
    print("weights = [(y, x) for x, y in zip(sizes[:-1], sizes[1:])] ->",
          [(y, x) for x, y in zip(sizes[:-1], sizes[1:])])
    print("b, w in zip(network.biases, network.weights) =", list(zip(network.biases, network.weights)))


if __name__ == "__main__":
    main()
