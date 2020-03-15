class LinearRegression():
    def __init__(self):
        print('Constructor of Linear Regression')
        
        # Variables to store final weights and bias
        self.final_weights = 0
        self.final_bias = 0
        
        # Cost history
        self.cost = []
    
    def initialize(self, num_features):
        return np.zeros((num_features, 1)), np.zeros((1))
    
    # mean squared error for objective function
    def calculate_cost(self, y_hat, y, batch_size):
        return  np.sum(np.square(y_hat - y)) / batch_size
    
    def feedforward_propagation(self, input_data, weights, bias):
        return np.dot(input_data, weights) + bias
    
    def gradient_descent(self, input_data, y, y_hat, weights, bias, learning_rate):
        dw = self.calculate_gradient_w(y_hat, y, input_data, self.num_records)
        db = self.calculate_gradient_b(y_hat, y, self.num_records)
        
        weights, bias = self.backward_propagation(y_hat, y, learning_rate, weights, bias, dw, db)
        return weights, bias
    
    def calculate_gradient_w(self, y_hat, y, input_data, batch_size):
        return np.sum(np.multiply((y_hat - y), input_data), axis = 0).reshape(self.num_features,1) / batch_size
    
    def calculate_gradient_b(self, y_hat, y, batch_size):
        return np.sum(y_hat - y, axis = 0).reshape(1,1) / batch_size
    
    def backward_propagation(self, y_hat, y, learning_rate, weights, bias, dw, db):
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
        return weights, bias
    
    def fit(self, input_data, y, num_iterations, learning_rate):
        
        # intialize dimensions
        self.num_records = input_data.shape[0]
        self.num_features = input_data.shape[1]
        batch_size = self.num_features
        
        # convert data into an operable format
        input_data = input_data.values
        y = y.values
        
        weights, bias = self.initialize(self.num_features)
        
        for iteration in range(num_iterations):
            y_hat = self.feedforward_propagation(input_data, weights, bias)
            
            weights, bias = self.gradient_descent(input_data, y, y_hat, weights, bias, learning_rate)
            
            individual_cost = self.calculate_cost(y_hat, y, batch_size)
            self.cost.append(individual_cost)
            
            print('cost =', individual_cost)
        
        self.final_weights, self.final_bias = weights, bias
        
    def predict(self, input_data):
        input_data = input_data.values
        y_hat = self.feedforward_propagation(input_data, self.final_weights, self.final_bias)
        return y_hat
    
    def plot_cost(self):
        indexes = list(range(len(self.cost)))
        
        axes = plt.gca()
        axes.set_xlim([0, len(self.cost) + 10])
        axes.set_ylim([0, max(self.cost) + 10])
        
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        
        plt.plot(indexes, self.cost)
        plt.show()
    
    def performance(self, y_hat, y, type_score):
        
        # convert data into an operable format
        y_hat = np.array(y_hat)
        y = np.array(y)
        
        # define length of test set
        num_records = y_hat.shape[0]
        if type_score == 'mae':
            return np.sum(np.abs(y_hat - y)) / num_records
