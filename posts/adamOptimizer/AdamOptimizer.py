import torch

class AdamOptimizer:
    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha, beta_1, beta_2, w_0 = None):

        # follow stochastic grad descent to get batches
        if(w_0 == None):
            w_0 = torch.rand(X.size()[1], requires_grad=True)
            
        # noise value
        epsilon = 10e-8

        self.model.w = w_0

        m_prev = torch.zeros(X.size()[1]) # first moment vector
        v_prev = torch.zeros(X.size()[1]) # second moment vector
        t = 0 #initialize timestep

        # adam optimization loop taken from pseudocode 
        for _ in range(1000): # check if gradient is really small or just for a nunber of iterations
            t += 1 # increment timestep
            g = self.model.grad(X, y) # get gradient at timestep t
            m = beta_1 * m_prev + (1 - beta_1) * g # update biased first moment estimate
            v = beta_2 * v_prev + (1 - beta_2) * g * g # update biased second raw moment estimate
            m_ = m/(1 - beta_1) # Compute bias-corrected first moment estimate
            v_ = v/(1 - beta_2) # Compute bias-corrected second raw moment estimate 
            self.model.w = self.model.w - alpha * m_/(torch.sqrt(v_) + epsilon) # update parameters



