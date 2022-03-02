import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import pickle
import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


class Regressor(nn.Module):
    def __init__(self, x=None, number_of_epochs=400, size_of_batches=128, learn_rate=0.01):
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
        # X, _ = self._preprocessor(x, training=True)
        self.input_size = 12
        self.hidden_layer_2 = 64
        self.hidden_layer_3 = 16
        self.output_size = 1
        self.number_of_epochs = number_of_epochs
        self.size_of_batches = size_of_batches
        self.learn_rate = learn_rate
        self.param_grid = {
            "number_of_epochs": [200, 250, 300],
            "learn_rate": [0.1, 0.01, 0.001],
            "size_of_batches": [64, 128, 256],
        }

        # sample for the model that we want to create
        self.model = None

        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def instantiate_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer_2),
            nn.Dropout(p=0),  # i.e. dropout currently doing nothing.
            nn.ReLU(),
            nn.Linear(self.hidden_layer_2, self.hidden_layer_3),
            nn.Dropout(p=0),  # also doing nothing here
            nn.ReLU(),
            nn.Linear(self.hidden_layer_3, self.output_size)
        )


    def shuffle_data(self, x, y, random_generator=default_rng()):
        shuffled_indices = random_generator.permutation(len(x))
        x_shuffled = x[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        return x_shuffled, y_shuffled

    """def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits"""

    # get_params method implemented in estimator to make gridsearchCV function
    def get_params(self, deep=True):
        return {
            "number_of_epochs": self.number_of_epochs, 
            "learn_rate": self.learn_rate, 
            "size_of_batches": self.size_of_batches
        }

    #set params method for gridsearchCV function
    def set_params(self, number_of_epochs, size_of_batches, learn_rate):
        self.number_of_epochs = number_of_epochs
        self.size_of_batches = size_of_batches
        self.learn_rate = learn_rate


    def _preprocessor(self, x, y=None, training=False):
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
        # print("Just entering processor, x shape: ", x.shape)

        # This filters out certain columns in Score, but not in predict.
        if y is not None:
            # merge the two, drop the outlying house values
            # print("Just entering processor, y shape: ", y.shape)
            combined = pd.merge(x, y, left_index=True, right_index=True)
            combined = combined[combined["median_house_value"] != 500001]

            # create our x and y again
            x = combined.drop(columns=["median_house_value"])
            y = pd.DataFrame(combined["median_house_value"])
            # print("in processor, y shape: ", y.shape)
            # print("in processor, x shape: ", x.shape)

        # We're going to fill the nan values with the average value of the column
        x = x.fillna(x.mean())

        # getting the per-household values as these are more insightful
        x["total_rooms"] = x["total_rooms"] / x["households"]
        x["total_bedrooms"] = x["total_bedrooms"] / x["households"]
        x["population"] = x["population"] / x["households"]

        # We believe that the households column confers no useful information for the model
        # so we're dropping it, having used it to get per-household figures elsewhere
        x.drop(columns=["households"], inplace=True)

        x.rename(
            columns={
                "total_rooms": "rooms_per_household",
                "total_bedrooms": "bedrooms_per_household",
                "population": "residents_per_household",
            },
            inplace=True,
        )

        # add one hot encoding to allow the model to work with text categories
        one_hot_area = pd.get_dummies(x["ocean_proximity"], prefix="area")
        x = x.drop(["ocean_proximity"], axis=1)

        # Scale x
        if training:
            x = self.x_scaler.fit_transform(x)
        else:
            x = self.x_scaler.transform(x)
        # print("in processor after done, x shape: ", x.shape)

        # Scale y
        if y is not None and training:
            y = self.y_scaler.fit_transform(y)
            # print("in processor after done, y shape: ", y.shape)

        elif y is not None:
            y = self.y_scaler.transform(y)
            # print("in processor after done, y shape: ", y.shape)

        x = pd.DataFrame(x)
        one_hot_area.reset_index(inplace=True)
        one_hot_area = one_hot_area.drop(["index"], axis=1)

        x = pd.merge(x, one_hot_area, left_index=True, right_index=True)
        # print("after merge: ", x.shape)
        x = x.to_numpy()
        # print("in processor after concatenation, x shape: ", x.shape)

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

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
        # print("Type of X: ", type(X), ": and shape: ", X.shape)
        # print("Type of Y: ", type(Y), ": and shape: ", Y.shape)
        # I think we should split our data into batches here

        # set input size of neural network
        self.input_size = X.shape[1]
        self.instantiate_model()


        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learn_rate)
        scheduler = ExponentialLR(optimiser, gamma=0.975)  # for lr decay
        criterion = torch.nn.MSELoss()

        # shuffle seed
        seed = 60012
        rg = default_rng(seed)
        X, Y = self.shuffle_data(X, Y, random_generator=rg)

        # shuffle split the dataset into specific number of batches
        # X, Y = self.shuffle_data(X, Y, random_generator=rg)

        # print("Shape: ", X.shape, Y.shape, "Type: ", type(X))
        number_of_batches = 0
        if len(X) <= self.size_of_batches:
            number_of_batches = 1
        else:
            number_of_batches = len(X) // self.size_of_batches
        # print("Number of batches: ", number_of_batches)
        X_batches = np.array_split(X, number_of_batches)
        Y_batches = np.array_split(Y, number_of_batches)

        # convert to torch tensors
        for batch_no in range(number_of_batches):
            X_batches[batch_no] = torch.from_numpy(X_batches[batch_no]).float()
            Y_batches[batch_no] = torch.from_numpy(Y_batches[batch_no]).float()
        # print("Shape: ", X_batches[0].shape, X_batches[1].shape, "Type: ", type(X_batches[0]))

        for epoch in range(self.number_of_epochs):
            for batch_no in range(number_of_batches):
                # Reset the gradients
                optimiser.zero_grad()

                # forward pass
                y_hat = self.model(X_batches[batch_no])

                # compute loss
                loss = criterion(y_hat, Y_batches[batch_no])

                # Backward pass (compute the gradients)
                loss.backward()

                # update parameters
                optimiser.step()

            scheduler.step()  # this is for the learning rate decay

            if epoch % 10 == 0:
                print(f"L: {loss:.4f}")
                #print(f"LR: {(scheduler.get_lr())}")

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
        X, _ = self._preprocessor(x, training=False)  # Do not forget
        X = torch.from_numpy(X).float()

        normalised_y_predictions = self.model(X)

        # denormalise the predictions, to get something useful for comparisons
        y_predictions = self.y_scaler.inverse_transform(
            normalised_y_predictions.detach().numpy()
        )

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
        print("Score function, entering processor:")
        # _, Y = self._preprocessor(x, y = y, training = False) # Do not forget

        print("x shape before processing in score: ", x.shape)
        print("y shape before processing in score: ", y.shape)
        y_hat = pd.DataFrame(self.predict(x))

        y_predictions = pd.merge(y_hat, y, left_index=True, right_index=True)
        y_predictions.columns = ["predicted", "gold"]
        y_predictions["difference"] = (
            y_predictions["gold"] - y_predictions["predicted"]
        ).apply(abs)
        print(y_predictions[["gold", "predicted"]])

        return mean_squared_error(y, y_hat) ** 0.5

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open("part2_model.pickle", "wb") as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open("part2_model.pickle", "rb") as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train, y_train):
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

    # This is the basic format of the gridsearchcv that I found on their 
    # tutorial page, here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html.
    regressor = Regressor(x_train)

    hyperparameter_tuner = GridSearchCV(regressor, regressor.param_grid, cv=5)
    hyperparameter_tuner.fit(x_train, y_train)
    print(type(hyperparameter_tuner))
    #hyperparameter_tuner.fit(x_train, y_train)

    print(sorted(hyperparameter_tuner.cv_results_))

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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    # Create the regressor model
    regressor = Regressor(x_train, number_of_epochs=200)

    # fit the model based on our held out training set
    regressor.fit(x_train, y_train)
    # save it for later
    save_regressor(regressor)
    # regressor = load_regressor()

    # Error
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

    RegressorHyperParameterSearch(x_train, y_train)


if __name__ == "__main__":
    example_main()
