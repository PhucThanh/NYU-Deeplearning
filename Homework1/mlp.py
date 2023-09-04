import torch


def activation(function, x):
    if function == "relu":
        return torch.relu(x)
    if function =="sigmoid":
        return torch.sigmoid(x)
    return x


def activation_deriative(function, x):
    if function == "relu":
        return (x>0).type(x.dtype)
    if function =="sigmoid":
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))
    return torch.ones_like(x)

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        self.cache["x"] = x
        self.cache["z1"] = torch.matmul(x, self.parameters["W1"].T) + self.parameters["b1"]
        self.cache["z2"] = activation(self.f_function, self.cache["z1"])

        self.cache["z3"] = torch.matmul(self.cache["z2"], self.parameters["W2"].T) + self.parameters["b2"]
        y_hat = activation(self.g_function, self.cache["z3"])

        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        
        dyhat_dz3 = activation_deriative(self.g_function, self.cache["z3"]) # (batch,linear_2_out_features)

        # Upstream to z3
        dJd_z3 = dJdy_hat * dyhat_dz3 # (batch, linear_2_out_features)

        # Local z3
        dz3_db2 = torch.ones(self.cache["z3"].shape[0]) # (batch,1) # Looks like shape depends on the deriative
        dz3_dW2 =  self.cache["z2"] # (batch, linear_2_in_features)

        # b2
        dJ_db2 = dJd_z3.T @ dz3_db2 # (batch, linear_2_out_features).T @ (batch,1) = (linear_2_out_features)
        self.grads["dJdb2"] = dJ_db2 

        # W2
        dJ_dW2 = dJd_z3.T @ dz3_dW2 # (batch, linear_2_out_features).T @ (batch, linear_2_in_features) = (linear_2_out_features, linear_2_in_features)
        self.grads["dJdW2"] = dJ_dW2

        # Upstream to z2
        dz3_dz2 = self.parameters["W2"] # (linear_2_out_features, linear_2_in_features)
        dJd_dz2 = dJd_z3 @ dz3_dz2 # (batch, linear_1_out_features)

        dz2_dz1 = activation_deriative(self.f_function, self.cache["z1"])
        dJ_dz1 = dJd_dz2 * dz2_dz1 # batch, linear_1_out_features

        dz1_db1 = torch.ones(self.cache["z1"].shape[0]) # (batch,)
        dz1_dW1 = self.cache["x"] # (batch, linear_1_in_features)

        dJ_db1 = dJ_dz1.T @ dz1_db1
        self.grads["dJdb1"] = dJ_db1

        dJ_dW1 = dJ_dz1.T @ dz1_dW1
        self.grads["dJdW1"] = dJ_dW1
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss = torch.mean((y - y_hat)**2)
    dJdy_hat = -2 * (y-y_hat) / (y.shape[0] * y.shape[1])

    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    # loss = -torch.mean((1-y) * torch.log(1-y_hat) + y * torch.log(y_hat))
    loss = -torch.mean(
        y*torch.log(y_hat) + 
        (1-y)*torch.log(1-y_hat)
        )
    norm_const = y.shape[0]*y.shape[1]
    dJdy_hat = (-(y/y_hat) + ((1-y)/(1-y_hat)))/norm_const
    return loss, dJdy_hat
