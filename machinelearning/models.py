import nn
from time import time
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x,self.w)
        

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        prediction=nn.as_scalar(self.run(x))
        if prediction>=0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        
        presence_erreur=True
        
        while presence_erreur:
            
            presence_erreur=False
        
            for x, y in dataset.iterate_once(1):
                if  self.get_prediction(x)!=nn.as_scalar(y):
                    presence_erreur=True
                      
                    self.w.update(x,nn.as_scalar(y))      
                


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # we have chosen 256 as a dimension for our matricies to make use of the binary nature of storage and to fit our matricies on cache
        self.W1 = nn.Parameter(1, 256)
        self.b1 = nn.Parameter(1, 256)
        
        self.W2 = nn.Parameter(256, 256)
        self.b2 = nn.Parameter(1, 256)
        
        self.W3 = nn.Parameter(256, 1)
        self.b3 = nn.Parameter(1, 1)
    
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        def couche(x,W,b):
            xm = nn.Linear(x, W)
            xm=nn.ReLU(nn.AddBias(xm, b))
            return xm
        # we have chosen 2 non linearities as it seemed to be a minimal number for our graph to fit correctly to the sin function
        x=couche(x,self.W1,self.b1)
        x=couche(x,self.W2,self.b2)
        x = nn.Linear(x, self.W3)
        x=nn.AddBias(x, self.b3)
        
        return nn.AddBias(x, self.b3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predicted_y=self.run(x)
        loss = nn.SquareLoss(predicted_y, y)
        return loss
    
    def train(self, dataset):
        """
        Trains the model.
        """

        value_final=float("inf")
        
        while value_final>0.015:
        
            for x, y in dataset.iterate_once(1):
    
                loss=self.get_loss(x,y)
                multiplier=-0.01
                grad_wrt_W1,grad_wrt_W2,grad_wrt_W3, grad_wrt_b1,grad_wrt_b2,grad_wrt_b3 = nn.gradients(loss, [self.W1,self.W2,self.W3,self.b1,self.b2,self.b3])
                
                self.W1.update(grad_wrt_W1, multiplier)
                self.W2.update(grad_wrt_W2, multiplier)
                self.b1.update(grad_wrt_b1, multiplier)
                self.b2.update(grad_wrt_b2, multiplier)
                self.W3.update(grad_wrt_W3, multiplier)
                self.b3.update(grad_wrt_b3, multiplier)
            value_final=nn.as_scalar(loss)
        
        
          

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size=32 # always for cache efficiency we have chosen a power of 2 as a batch_size althouth we have no theoretical ground for this
        #value it proved after some testing to get faster convergence because of how matrix multiplications are handled
        
        self.W1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        
        self.W2 = nn.Parameter(256, 256)
        self.b2 = nn.Parameter(1, 256)
        
        self.W4 = nn.Parameter(256, 256)
        self.b4 = nn.Parameter(1, 256)
        
        self.W3 = nn.Parameter(256, 10)
        self.b3 = nn.Parameter(1, 10)
        


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        def couche(x,W,b):
            x = nn.Linear(x, W)
            x=nn.ReLU(nn.AddBias(x, b))
            return x
        
        x=couche(x,self.W1,self.b1)
        x=couche(x,self.W2,self.b2)
        x=couche(x,self.W4,self.b4)
        x = nn.Linear(x, self.W3)
        
        
        return nn.AddBias(x, self.b3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y=self.run(x)
        loss = nn.SoftmaxLoss(predicted_y, y)
        return loss
        

    def train(self, dataset):
        """
        Trains the model.
        """
        
        value_final=0
        while value_final<0.974:
        
            for x, y in dataset.iterate_once(self.batch_size):
                
    
                loss=self.get_loss(x,y)
                multiplier=-0.01
                grad_wrt_W1,grad_wrt_W2,grad_wrt_W3, grad_wrt_b1,grad_wrt_b2,grad_wrt_b3,grad_wrt_W4,grad_wrt_b4 = nn.gradients(loss, [self.W1,self.W2,self.W3,self.b1,self.b2,self.b3,self.W4,self.b4])
                
                self.W1.update(grad_wrt_W1, multiplier)
                self.W2.update(grad_wrt_W2, multiplier)
                self.b1.update(grad_wrt_b1, multiplier)
                self.b2.update(grad_wrt_b2, multiplier)
                self.W3.update(grad_wrt_W3, multiplier)
                self.b3.update(grad_wrt_b3, multiplier)
                self.W4.update(grad_wrt_W4, multiplier)
                self.b4.update(grad_wrt_b4, multiplier)
                
            value_final=dataset.get_validation_accuracy()
        
            

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W = nn.Parameter(47, 256)
                
        self.W_hidden = nn.Parameter(256, 256)
        
        self.W_langue = nn.Parameter(256,5)
        
        
    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        h=nn.ReLU(nn.Linear(xs[0], self.W))
        if len(xs)>1:
            
            for ith_lettre in xs[1:]:
                h = nn.ReLU(nn.Add(nn.Linear(ith_lettre, self.W), nn.Linear(h, self.W_hidden)))
                
        return nn.Linear(h,self.W_langue)
            

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y=self.run(xs)
        loss = nn.SoftmaxLoss(predicted_y, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        
        value_final=0
        while value_final<0.85:
        
            for xs, y in dataset.iterate_once(4):
    
                loss=self.get_loss(xs,y)
                multiplier=-0.05
                grad_wrt_W,grad_wrt_W_langue,grad_wrt_W_hidden = nn.gradients(loss, [self.W,self.W_langue,self.W_hidden])
                
                self.W.update(grad_wrt_W, multiplier)
                self.W_langue.update(grad_wrt_W_langue, multiplier)
                
                self.W_hidden.update(grad_wrt_W_hidden, multiplier)
                
            value_final=dataset.get_validation_accuracy()
        
