import torch

class LinearModel:
    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        return X@self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        s = X@self.w
        return (s > 0).int()

class LogisticRegression(LinearModel):
    
    def loss(self, X, y):
        """
        Compute empirical risk using logistic loss function

        Arguments:
            X, torch.Tensor: the feature matrix. X.size() = (n,p),
            where n is the number of data points and p is the dimension
            of features. This implementation assumes the final column
            of X is all 1s.

            y, torch.Tensor: the target vector. y.size() = (n, 1),
            where n is the number of data points. y is assumed to be 
            in the range of {0, 1}.

        Returns:
            L(w), torch.Tensor: the loss of the model at the current weight value
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        s = X@self.w
        # print(f'w: {self.w}')
        sig = 1/(1+torch.exp(-s))
        # print(f'y size {y.size()}')
        # print(f'sig size {sig.size()}')
        # print(f'sig log size {sig.log().size()}')
        return -(y*sig.log() + (1 - y)*(1-sig).log()).mean()


    def grad(self, X, y):
        """
        Compute the gradient of the empirical risk

        Arguments:
            X, torch.Tensor: the feature matrix. X.size() = (n,p),
            where n is the number of data points and p is the dimension
            of features. This implementation assumes the final column
            of X is all 1s.

            y, torch.Tensor: the target vector. y.size() = (n, 1),
            where n is the number of data points. y is assumed to be 
            in the range of {0, 1}.

        Returns:
            grad L(w), torch.Tensor: the gradient of the empirical risk
        """
        s = X@self.w
        sig = 1/(1+torch.exp(-s))
        v = (sig-y)
        v_ = v[:, None]
        return torch.mean((v_*X), 0)

    def hessian(self, X):
        """
        Compute the hessian matrix using the supplied data

        Arguments:


        Returns:
            Hessian matrix
        """

        s = X@self.w
        sig = 1/(1+torch.exp(-s))
        sig_matrix = sig*(1-sig)
        D = torch.diag(sig_matrix)
        return torch.t(X) @ D @ X
        
class NewtonOptimizer:

    def __init__(self, model):
        self.model = model
    
    def step(self, X, y, alpha):
        self.model.w = self.model.w - alpha*torch.inverse(self.model.hessian(X))@self.model.grad(X, y)

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha, beta, w_prev):
        """
        Gradient descent with momentum

        Arguments:
            X, torch.Tensor: the feature matrix. X.size() = (n,p),
            where n is the number of data points and p is the dimension
            of features. This implementation assumes the final column
            of X is all 1s.

            y, torch.Tensor: the target vector. y.size() = (n, 1),
            where n is the number of data points. y is assumed to be 
            in the range of {0, 1}.

            alpha, float: the learning rate of the model should be 
            between {0,1}

            beta, float: the learning rate of the momentum term, if 0
            momentum is not used. Recommended value for momentum is 0.9

        """
        self.model.w = self.model.w - alpha*self.model.grad(X, y) + beta*(self.model.w - w_prev)





