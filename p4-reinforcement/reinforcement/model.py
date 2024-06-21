import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim
        
        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        
        # Setting hyperparameters
        self.learning_rate = 0.5
        self.numTrainingGames = 6000
        self.batch_size = 32

        # Initialize neural network layers as Parameters
        self.couche_initial = nn.Parameter(self.state_size, 256)  
        self.couche_cache1 = nn.Parameter(256,256) 
        self.couche_cache2=nn.Parameter(256, self.num_actions) 
    
        self.parameters = [self.couche_initial,self.couche_cache1,self.couche_cache2]
        
    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        estimatedQ = self.run(states)
        return nn.SquareLoss(estimatedQ, Q_target)  

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        
        def layering(x,w,relu=True):
            # This function creates a neural network layer. It applies a linear transformation
            # followed optionally by a ReLU activation, depending on the `relu` argument.
                      
            if not relu:
               return nn.Linear(x,w) 
            return nn.ReLU(nn.Linear(x,w))
        
        # three layers network
        first_layer=layering(states,self.couche_initial)
        second_layer=layering(first_layer,self.couche_cache1)
        
        # Apply the third layer without ReLU activation (relu=False).
        return layering(second_layer,self.couche_cache2,relu=False)
        
           
    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        
        loss = self.get_loss(states,Q_target)
        gradients = nn.gradients(loss, self.parameters)

        # Loop over pairs of parameters and their corresponding gradients
        for parameter,direction in zip(self.parameters,gradients):
            # Update each parameter in the direction that minimizes the loss, scaled by the learning rate.
            parameter.update(direction,-self.learning_rate)
            
