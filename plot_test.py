import matplotlib.pyplot as plt
import part1_nn_lib as nn
import numpy as np

def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "identity"]
    net = nn.MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")

    ######
    feature_names = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    print(type(dat))

    
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
    xx = list(range(0, len(trainer.losses)))
    yy = trainer.losses
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax[1].plot(xx, yy)
    plt.show()


if __name__ == "__main__":
    example_main()
