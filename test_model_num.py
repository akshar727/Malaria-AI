batch_sizes = [32, 64]
conv_layers = [3, 4, 5]
layer_sizes = [32, 64, 128, 256]
dense_layers = [0, 1, 2, 3, 4, 5]


def get_model_number(batch_size, conv_layer, layer_size, dense_layer):
    batch_size_index = batch_sizes.index(batch_size)
    conv_layer_index = conv_layers.index(conv_layer)
    layer_size_index = layer_sizes.index(layer_size)
    dense_layer_index = dense_layers.index(dense_layer)

    return (batch_size_index * len(conv_layers) * len(layer_sizes) * len(dense_layers) +
            conv_layer_index * len(layer_sizes) * len(dense_layers) +
            layer_size_index * len(dense_layers) +
            dense_layer_index + 1)



for batch_size in batch_sizes:
    for conv_layer in conv_layers:
        for layer_size in layer_sizes:
            for dense_layer in dense_layers:
                print(get_model_number(batch_size, conv_layer, layer_size, dense_layer))