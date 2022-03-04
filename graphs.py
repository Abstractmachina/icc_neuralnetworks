import matplotlib.pyplot as plt
import part1_nn_lib as nn
import numpy as np

from scipy.interpolate import make_interp_spline, BSpline
import pandas as pd
import json

#BG_COLOR = (60.0/255.0, 110.0/255.0, 185.0/255.0, 1.0)
BG_COLOR = "white"
GRAPH_COLOR = "white"

class PlotDataCollector:
    def __init__(self):
        self.losses = list()
        self.trainingData = None
        self.trainingDataLabels= None
        self.rawData = None
        self.tuningResults = pd.DataFrame()
        self.resultsFileName = "results"

    #data collection
    def addLoss(self, loss):
        self.losses.append(loss)

    def addTrainingData(self, x, labels = None):
        self.trainingData = x
        if labels != None:
            self.trainingDataLabels = labels

    def addRawData(self, x):
        self.rawData = x


    def setResultFileName(self, filename):
        self.resultsFileName = filename

    @staticmethod
    def saveTuningResults(result, filename):
        try:
            with open(filename,'r') as file:
                file_data = json.load(file)
                
            file_data.append(result)
            # Sets file's current position at offset.
            #file.seek(0)
            # convert back to json.
            with open(filename, "w") as file:
                json.dump(file_data, file)
        except FileNotFoundError:
            results = list()
            results.append(result)
            with open(filename, 'w') as file:
                json.dump(results, file)
        #self.tuningResults.append(result, ignore_index=True)
        return

    @staticmethod
    def loadResults(filename):
        with open(filename,'r') as file:
            file_data = json.load(file)
        return file_data


    #plotting
    def plotTrainingData(self):
        self.boxPlot(self.trainingData, tickLabels=self.trainingDataLabels)

    def plotLoss(self):
        x = range(len(self.losses))
        #self.simpleGraph(np.array(x), np.array(self.losses), title = "Loss Over Epoch", xLabel="Epoch", yLabel = "Loss")
        self.graph_fitted(np.array(x), np.array(self.losses))

    def plotAccuracy(self):
        '''
        save accuracy per trained model. Use for hyperparamter tuning.
        '''

    #================================================================================
    #graph types
    @staticmethod
    def simpleGraph(x_dat, y_dat, title = "title", xLabel = "x", yLabel = "y"):
        '''
        Arguments:
        -x_dat, y_dat -- list of values. can be list or np.array
        '''
        plt.figure()
        ax = plt.axes()
        ax.set_facecolor(BG_COLOR)
        #plt.rc('grid', linestyle=':', color='red', linewidth=2)
        plt.rc("grid", linewidth = 0.05)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.plot(x_dat, y_dat, marker='o', ms = 0.1)
        plt.grid(True)
        plt.show()
        return

    @staticmethod
    def boxPlot(dat, title = "title", xLabel = "x", yLabel = "y", tickLabels = None, padding = 0.5, notch = False):

        #y_bounds = (np.min(dat), np.max(dat))

        plt.figure()
        plt.boxplot(dat, notch=notch, widths=0.5, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 0.5})
        plt.grid(axis='y')

        if tickLabels != None:
            tickCount = range(1, len(tickLabels)+1)
            plt.xticks(tickCount, tickLabels, rotation=90)
        '''
        plt.boxplot(dat, notch=notch, widths=1, patch_artist=True,
                    showmeans=False, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                            "linewidth": 0.5},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5})
        ax = plt.axes()
        '''
        #ax.set_facecolor(BG_COLOR)
        #plt.set(xlim=(0, dat.shape[1]+padding), xticks=np.arange(0, dat.shape[1]+padding),
        #    ylim=(y_bounds[0]-padding, y_bounds[1]+padding), yticks=np.arange(y_bounds[0]-padding, y_bounds[1]+padding))
        
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        plt.show()
        return

    @staticmethod
    def boxPlot_pd(df):
        '''
        Arguments:
        -df -- pandas dataframe
        '''
        df.boxplot(rot = 90, fontsize = 10)
        plt.show()
    @staticmethod
    def histogram(dat, title = "title", xLabel = "x", yLabel = "y"):
        plt.figure()
        plt.hist(dat)

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        ax = plt.axes()
        ax.set_facecolor("blue")
        plt.show
        return

    @staticmethod
    def scatterPlot(x_dat, y_dat, title = "title", xLabel = "x", yLabel = "y"):
        plt.figure()

        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

        plt.scatter(x_dat, y_dat, s = 0.1)
        plt.show()
        return

    @staticmethod
    def graph_fitted(x, y, title = None, xLabel = None, yLabel = None):
        plt.figure()

        plt.plot(x, y, marker='o', ms = 0.1)
        
        """
        xnew = np.linspace(x.min(), x.max(), 100) 
        spl = make_interp_spline(x, y, k=1)
        y_smooth = spl(xnew)
        plt.plot(xnew, y_smooth)
        """

        df = pd.DataFrame({"x": x,"y": y})
        df['Rolling'] = df['y'].rolling(7, min_periods = 3).mean()

        print(type(df['Rolling']))
        xnew = np.linspace(df['Rolling'].min(), df['Rolling'].max(), 100) 
        spl = make_interp_spline(x, df['Rolling'], k=3)
        y_smooth = spl(xnew)
        plt.plot(xnew, y_smooth)

        #plt.plot(x, df['Rolling'])
        #print(df)

        plt.rc("grid", linewidth = 0.05)
        plt.grid(True)
        if title != None:
            plt.title(title)
        if xLabel != None:
            plt.xlabel(xLabel)
        if yLabel != None:
            plt.ylabel(yLabel)
        
        plt.show()

        
        return


#===============================================================
#===============================================================


def example_main():
    
    #copied part1 example for testing
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = nn.MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")

    ######
    feature_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    print(dat)

    boxPlot(dat, notch=True)

    x_dat = np.array(range(10))
    y_dat = np.array(range(10))
    simpleGraph(x_dat, y_dat, "Epoch", "Loss")



    return
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = nn.Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = nn.Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="bce",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

    #box plot
    #plt.style.use('_mpl-gallery')

    fig, ax = plt.subplots(2, 1)
    #VP = ax.boxplot(dat)
    y_bounds = (np.min(dat), np.max(dat))
    padding = 0.5
    ax[0].boxplot(dat,  widths=1, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})

    ax[0].set(xlim=(0, dat.shape[1]+padding), xticks=np.arange(0, dat.shape[1]+padding),
        ylim=(y_bounds[0]-padding, y_bounds[1]+padding), yticks=np.arange(y_bounds[0]-padding, y_bounds[1]+padding))
    plt.xlabel('feature')
    plt.ylabel('value')
    #plt.show()
    
    #scatter plot
    '''
    plt.figure()
    plt.scatter(dat[:,0], dat[:,1], c=dat[:,-1], cmap=plt.cm.Set1, edgecolor='k')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.show()
    '''
    #####
    #plot losses
    #xx = list(range(0, len(trainer.losses)))
    #yy = trainer.losses
    


if __name__ == "__main__":
    example_main()
