print("Hello, World!")


inputs = [0, 0, 1, 0]
weights = [0, 0, 0, 0]
desired_result = 1
learning_rate = 0.2
trials = 6

def evaluate_neural_network(input_array, weight_array):
    result = 0
    for i in range(len(input_array)):
        layer_value = input_array[i] * weight_array[i]
        result += layer_value
    print("evaluate_neural_network: " + str(result))
    print("weights: " + str(weights))
    return result

def evaluate_error(desired, actual):
    error = desired - actual
    print("evaluate_error: " + str(error))
    return error

def learn(input_array, weight_array):
    print("learning... please wait")
    for i in range(len(input_array)):
        if input_array[i]>0:
            weight_array[i] += learning_rate
            
def train(trials):
    for i in range(trials):
        neural_net_result = evaluate_neural_network(inputs, weights)
        learn(inputs, weights)
        
train(trials)

