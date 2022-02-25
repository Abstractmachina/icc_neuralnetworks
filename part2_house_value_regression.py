import torch
from torch import nn, tensor
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class Regressor(nn.Module):

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        super().__init__()

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.number_of_epochs = nb_epoch

        # sample for the model that we want to create
        self.model = nn.Sequential(
            nn.Linear(self.input_size, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, self.output_size)
        )


        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def _preprocessor(self, x, y = None, training = False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size).
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if y is not None:
            #merge the two, drop the outlying house values
            combined = pd.merge(x, y, left_index=True, right_index=True)
            combined = combined[combined["median_house_value"] != 500001] 

            # create our x and y again
            x = combined.drop(columns=["median_house_value"])
            y = pd.DataFrame(combined["median_house_value"])


        # add one hot encoding to allow the model to work with text categories
        one_hot_area = pd.get_dummies(x['ocean_proximity'], prefix="area")
        
        x = pd.merge(x, one_hot_area, left_index=True, right_index=True)
        
        # We're going to fill the nan values with the average value of the column
        x = x.fillna(x.mean())
        
        # unnecessary column, we've one hot encoded it now
        x = x.drop(['ocean_proximity'], axis=1)

        # getting the per-household values as these are more insightful
        x["total_rooms"] = x["total_rooms"] / x["households"]
        x["total_bedrooms"] = x["total_bedrooms"] / x["households"]
        x["population"] = x["population"] / x["households"]


        # We believe that the households column confers no useful information for the model
        # so we're dropping it, having used it to get per-household figures elsewhere
        x.drop(columns=["households"], inplace=True)

        x.rename(columns = {"total_rooms":"rooms_per_household", 
                "total_bedrooms":"bedrooms_per_household", 
                "population":"residents_per_household"}, inplace=True)
        
        # logic to separate out training data from test data and scale by
        # standardisation
        if training:
            x = self.x_scaler.fit_transform(x)
        else:
            x = self.x_scaler.transform(x)

        if y is not None:
            y = self.y_scaler.fit_transform(y)
        
        # Return preprocessed x and y
        return x, y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        
        optimiser = torch.optim.SGD(self.model.parameters(), lr=0.0005)
        criterion = torch.nn.MSELoss()

        for epoch in range(self.number_of_epochs):
            # forward pass
            y_hat = self.model(X)
            # compute loss
            loss = criterion(y_hat, Y) 

            # Reset the gradients 
            optimiser.zero_grad() 

            # Backward pass (compute the gradients)
            loss.backward()

            # update parameters
            optimiser.step()

            print(f"L: {loss:.4f}")

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        X = torch.from_numpy(X).float()

        normalised_y_predictions = self.model(X)

        #denormalise the predictions, to get something useful for comparisons
        y_predictions = self.y_scaler.inverse_transform(normalised_y_predictions.detach().numpy())

        return y_predictions

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch():
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    x = data.loc[:, data.columns != output_label]
    y = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # Create the regressor model
    regressor = Regressor(x_train, nb_epoch = 5000)

    # fit the model based on our held out training set
    regressor.fit(x_train, y_train)
    # save it for later
    save_regressor(regressor)
    # regressor = load_regressor()

    # Predictions - type shifting is so annoying, we're forcing y_predictions from np.array to pd.DataFrame here
    # most of this is just me fiddling to try to get something resembling a metric
    y_predictions = pd.DataFrame(regressor.predict(x_test))
    y_predictions = pd.merge(y_predictions, y_test, left_index=True, right_index=True)
    y_predictions.columns = ["predicted", "gold"]
    y_predictions["difference"] = (y_predictions["gold"] - y_predictions["predicted"]).apply(abs)

    mean_error = y_predictions["difference"].mean()
    print("mean error is:", mean_error)
    print(y_predictions[["gold", "predicted"]])
    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
