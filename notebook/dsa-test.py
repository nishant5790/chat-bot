
# most frequent words implementation without using any external libraries
# This function takes a string input and returns the n most frequent words along with their counts.
import numpy as np
def most_frequent_words(text, n=10):
    # Split the text into words and convert to lowercase
    words = text.lower().split()
    
    # Create a dictionary to count word frequencies
    word_count = {}
    
    for word in words:
        # Remove punctuation from the word
        word = ''.join(char for char in word if char.isalnum())
        if word:  # Check if the word is not empty
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
    
    # Sort the dictionary by frequency and get the top n words
    sorted_words = sorted(word_count.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_words[:n]

# check balaned parentheses implementation without using any external libraries
def check_balanced_paraentheses(exp:str)-> bool:
    stack = []
    parentheses = {'(': ')', '{': '}', '[': ']'}
    
    for char in exp:
        if char in parentheses:  # If it's an opening bracket
            stack.append(char)
        elif char in parentheses.values():  # If it's a closing bracket
            if not stack or parentheses[stack.pop()] != char:
                return False
    
    return len(stack) == 0  # Return True if stack is empty, meaning all brackets are balanced

def shift_zeroes(arr):
    pass
    for ele in arr:
        if ele == 0 :
            arr.remove(ele)
            arr.append(ele)
    return arr

def find_prime_numbers(n):
    """Returns a list of prime numbers up to n."""
    primes = []
    for num in range(2, n + 1):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        self.hidden_layer_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_activation)
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        return self.sigmoid(self.output_layer_activation)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * (output * (1 - output))

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * (self.hidden_layer_output * (1 - self.hidden_layer_output))

        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')

#Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
def generate_parentheses(n):
    def generate(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        if open_count < n:
            generate(current + '(', open_count + 1, close_count)
        if close_count < open_count:
            generate(current + ')', open_count, close_count + 1)

    result = []
    generate('', 0, 0)
    return result

# function to find the fabbonaci serise without using stack or recursion
# This function returns the first n numbers in the Fibonacci series.
def fibonacci(n):
    fib_series = []
    a, b = 0, 1
    for _ in range(n):
        fib_series.append(a)
        a, b = b, a + b
    return fib_series


# Example usage
if __name__ == "__main__":
    text = "This is a test. This test is only a test."
    print(most_frequent_words(text, 3))  # Output: [('test', 3), ('this', 2), ('is', 2)]
    
    expression = "{[()()]}"
    print(check_balanced_paraentheses(expression))  # Output: True
    
    expression = "{[(])}"
    print(check_balanced_paraentheses(expression))  # Output: False
    arr = [0, 1, 0, 3, 12]
    print(shift_zeroes(arr))  # Output: [1, 3, 12, 0, 0]
    n = 30
    print(find_prime_numbers(n))  # Output: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


