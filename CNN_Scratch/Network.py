class Network:

    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self,loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):

        for epoch in range(epochs):
            print(epoch)
            err = 0
            for sample in  range(len(x_train)):
                print(sample)
                output = x_train[sample]
                for layer in self.layers:

                    output = layer.forward_propagation(output)

                err += self.loss(y_train[sample], output)
                error = self.loss_prime(y_train[sample], output)
                for layer in reversed(self.layers):

                    error = layer.backward_propagation(error, learning_rate)
            err /= len(x_train)
            print('epoch %d  error is %f' % (epoch,err))