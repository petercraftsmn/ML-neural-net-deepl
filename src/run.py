from src.network import Network


def main():
    sizes = [2, 3, 1]
    network = Network(sizes)
    print("num_layers")
    print(network.num_layers)
    print("sizes")
    print(sizes[1:])
    print(network.sizes)
    print("biases")
    print(network.biases)
    print("weight")
    print(network.weights)


if __name__ == "__main__":
    main()
