from pennylane import numpy as np
from sklearn import datasets as skl_data
import pandas as pd

import matplotlib.pyplot as plt
import os
import time
from functions_classes_nQuBit import make_lines,predictor,plot_predictions

plt.rcParams.update({
    "text.usetex": True,
    'text.latex.preamble': "\\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "axes.labelsize": 12,
    "xtick.labelsize": 12
})

# ---------------------------------------------------------------------------------------------------------------------------
#                                               generating the data
# ---------------------------------------------------------------------------------------------------------------------------

dataset = "LHCo"

if dataset == "sklblobs":
    points_all,label_all = skl_data.make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,2]],random_state=42,cluster_std=.7)
elif dataset == "sklmoons":
    points_all,label_all = skl_data.make_moons(n_samples=1000,noise=.1,random_state=42)
elif dataset == "sklcircles":
    points_all,label_all = skl_data.make_circles(n_samples=1000,noise=.1,random_state=42,factor=.5)
elif dataset == "lines":
    points_all,label_all = make_lines(d=2,l=10,phi=np.pi/4,n_samples=1000,noise=.3)
elif dataset == "LHCo":
    file = pd.read_hdf("../LHC_olympics/events_anomalydetection_v2.features.h5")
    # Data included:
    #       'pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2', 'label'
    #       3-momenta, invariant masses, and n-jettiness variables tau1, tau2 and tau3 for the highest pT jet (j1) and the second highest pT jet (j2)

    # data is sorted by label, so I'll shuffle:
    # file = file.sample(frac=1).reset_index(drop=True)

    def func(x):
        # print(x.shape[0])
        # x.info()
        return x.sample(1000)
            # pd.dataframe.sample(n) returns a random sample of n elements
        return x.sample(int(sum(file["label"])))

    # sampling such that we have an equal number of background and signal events in our training data (and shuffling again)
    sample = file.groupby("label",group_keys=False).apply(func=func).sample(frac=1).reset_index(drop=True)

    points_all = np.stack((# sample["mj1"].to_numpy(),
                           sample["mj2"].to_numpy(),
                           sample["tau1j1"].to_numpy(),
                           # sample["tau2j1"].to_numpy(),
                           # sample["tau3j1"].to_numpy(),
                           # sample["tau1j2"].to_numpy(),
                           # sample["tau2j2"].to_numpy(),
                           # sample["tau3j2"].to_numpy()
                          ),
                          axis=1
    )
    label_all = sample["label"].to_numpy()

    if False:    # Visualizing the dataset
        # preparation
        colors = ["red" if y else "blue" for y in label_all]
        x = {0:sample["mj1"].to_numpy(),1:sample["mj2"].to_numpy()}
        x_name = {0:"mj1",1:"mj2"}
        x_label = {0:"$m^{\\text{j1}}$",1:"$m^{\\text{j2}}$"}
        y = {0:sample["tau1j1"].to_numpy(),1:sample["tau2j1"].to_numpy(),2:sample["tau3j1"].to_numpy(),3:sample["tau1j2"].to_numpy(),4:sample["tau2j2"].to_numpy(),5:sample["tau3j2"].to_numpy()}
        y_name = {0:"tau1j1",1:"tau2j1",2:"tau3j1",3:"tau1j2",4:"tau2j2",5:"tau3j2"}
        y_label = {0:"$\\tau_1^{\\text{j1}}$",1:"$\\tau_2^{\\text{j1}}$",2:"$\\tau_3^{\\text{j1}}$",3:"$\\tau_1^{\\text{j2}}$",4:"$\\tau_2^{\\text{j2}}$",5:"$\\tau_3^{\\text{j2}}$"}
        # plotting
        fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(15*2/3,10*2/3),num="feature_combinations",constrained_layout=True)
        for row in range(2):
            for col in range(3):
                ax[row,col].scatter(x[row],y[col],c=colors,alpha=.1,marker=".")
                ax[row,col].set_xlabel(x_label[row])
                ax[row,col].set_ylabel(y_label[col])
        plt.savefig("../LHC_olympics/visualization/feature_combinations.pdf",bbox_inches="tight")

        # plotting signal and background individually:
        signal_index = (label_all == 1)
        background_index = (label_all == 0)

        for row in range(2):
            for col in range(3):
                fig,ax = plt.subplots(nrows=1,ncols=2,num=x_name[row] + "_vs_" + y_name[col],figsize=(10,5))
                # signal:
                ax[0].scatter(x[row][signal_index],y[col][signal_index],label="signal",alpha=.1,marker=".")
                ax[0].set_xlabel(x_label[row])
                ax[0].set_ylabel(y_label[col])
                ax[0].legend()
                ax[0].grid()
                # background:
                ax[1].scatter(x[row][background_index],y[col][background_index],label="background",alpha=.1,marker=".")
                ax[1].set_xlabel(x_label[row])
                ax[1].set_ylabel(y_label[col])
                ax[1].legend()
                ax[1].grid()
                plt.savefig("../LHC_olympics/visualization/" + x_name[row] + "_vs_" + y_name[col] +".pdf",bbox_inches="tight")

        # plt.show()
        plt.close("all")
        exit()

# ---------------------------------------------------------------------------------------------------------------------------
#                                               hyperparameters and feature pipeline
# ---------------------------------------------------------------------------------------------------------------------------

learning_rate = 1
        # around 1 for qml, around .05 for classical machine learning
implementation = "prerot"
        # "scratch" or "pennylane" or "classical" or "prerot" or "pretrans" or "pregalilei"
optimizer = "COBYLA"
        # "vanillaGD" or "COBYLA"
structure_string = "denseanglemodprepISINGent"
node_structure = [100,100,100,100]#[4,4,4,4]
n_layers = 4
n_QuBit = 4
information_density = 2

n_epochs = 2

# features:
if points_all.shape[-1] == 2:
    if n_QuBit*information_density == 2:
        features_all = points_all
    elif n_QuBit*information_density == 4:
        features_all = np.insert(arr=points_all,obj=2,values=(points_all[:,0],points_all[:,1]),axis=1)
    elif n_QuBit*information_density == 8:
        features_all = np.insert(arr=points_all,obj=2,values=(points_all[:,0],points_all[:,1],points_all[:,0],points_all[:,1],points_all[:,0],points_all[:,1]),axis=1)
elif points_all.shape[-1] == 4:
    if n_QuBit*information_density == 4:
        features_all = points_all
    elif n_QuBit*information_density == 8:
        features_all = np.insert(arr=points_all,obj=2,values=(points_all[:,0],points_all[:,1],points_all[:,2],points_all[:,3]),axis=1)
assert features_all.shape[-1] == n_QuBit*information_density

# training and validation data
val_frac = .5
X_train = features_all[int(features_all.shape[0]*val_frac):,:]
X_val   = features_all[:int(features_all.shape[0]*val_frac),:]
Y_train = label_all[int(label_all.shape[0]*val_frac):]
Y_val   = label_all[:int(label_all.shape[0]*val_frac)]

# values for normalization:
points_max = ()
points_min = ()
for i in range(n_QuBit*information_density):
    points_max += (max(features_all[:,i]),)
    points_min += (min(features_all[:,i]),)
# points_max += (max((points_min[0]-x_mean)*(points_min[1]-y_mean),(points_min[0]-x_mean)*(points_max[1]-y_mean),(points_max[0]-x_mean)*(points_min[1]-y_mean),(points_max[0]-x_mean)*(points_max[1]-y_mean)),)
# points_min += (min((points_min[0]-x_mean)*(points_min[1]-y_mean),(points_min[0]-x_mean)*(points_max[1]-y_mean),(points_max[0]-x_mean)*(points_min[1]-y_mean),(points_max[0]-x_mean)*(points_max[1]-y_mean)),)

# ---------------------------------------------------------------------------------------------------------------------------
#                                               creating the model
# ---------------------------------------------------------------------------------------------------------------------------

model = predictor(implementation=implementation,
                  n_QuBit = n_QuBit,
                  learning_rate=learning_rate,
                  node_structure=node_structure,
                  n_layers=n_layers,
                  optimizer=optimizer,
                  information_density=information_density)
model.normalize(min=points_min,max=points_max,min_norm=0,max_norm=2*np.pi)

# ---------------------------------------------------------------------------------------------------------------------------
#                                               training
# ---------------------------------------------------------------------------------------------------------------------------

# status update
print("\nSignal fraction in training and validation data:\n    train: {:.3f}\n    val:   {:.3f}\nNumbers of data points:\n    train: {}\n    val:   {}\n".format(sum(Y_train)/len(Y_train),sum(Y_val)/len(Y_val),len(Y_train),len(Y_val)) +
    "implementation = " + model.implementation + "\n" +
    "dataset        = " + dataset + "\n" +
    "optimizer      = " + model.optimizer + "\n" +
    "structure      = " + structure_string + "\n" +
    "n_QuBit        = {}".format(model.n_QuBit) + "\n" +
    "n_layers       = {}".format(model.n_layers) + "\n" +
    "n_epochs       = {}".format(n_epochs) + "\n" +
    "--------------------------\n     Training begins.     \n--------------------------\n"
)

save_dataset    = True if points_all.shape[-1] == 2 else False
save_epochs     = True if points_all.shape[-1] == 2 else False
save_results    = True
show_epochs     = False
show_results    = False
write_metrics   = True
write_gradients = True
write_params    = True

if save_epochs or save_results or write_metrics:    # creating a directory in which to store data:
    now = time.localtime()
    if model.implementation != "classical":
        dirname = "results/{}QuBit_{}Layer/".format(n_QuBit,model.n_layers) + dataset + "_" + implementation + "_" + structure_string + "_" + model.optimizer + "_" + time.strftime("%d_%m_%Y_%H_%M_%S",now)
    else:
        dirname = "results/{}QuBit_{}Layer/".format(n_QuBit,model.n_layers) + dataset + "_classical_"
        for node in model.node_structure:
            dirname += "{}d".format(node)
        dirname += ("_" + time.strftime("%d_%m_%Y_%H_%M_%S",now))
    os.makedirs(dirname,exist_ok=True)

if save_dataset:    # Plotting the dataset
    fig = plt.figure(num="dataset_plot",figsize=(15,15))
    colors = ["red" if y else "blue" for y in label_all]        # red for signal (y==1) and blue for background (y==0)
    if points_all.shape[0] > 600:
        plt.scatter(x=points_all[::int(points_all.shape[0]/600),0],y=points_all[::int(points_all.shape[0]/600),1],c=colors[::int(points_all.shape[0]/600)])
    else:
        plt.scatter(x=points_all[:,0],y=points_all[:,1],c=colors)
    plt.scatter(x=points_all[:,0],y=points_all[:,1],c=colors)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.savefig(dirname + "/dataset.pdf",bbox_inches="tight")
    # plt.show()
    plt.close("dataset_plot")

if save_epochs or show_epochs:      # preparing the area on which I'll plot the network predictions (of course only possible for 2-dimensional datasets):
    assert points_all.shape[-1] == 2, "too many or too little features to display in a two-dimensional plot"
    sidelength = 100
    x_min = min(points_all[:,0])
    x_max = max(points_all[:,0])
    y_min = min(points_all[:,1])
    y_max = max(points_all[:,1])

    x_range = np.linspace(start=x_min,stop=x_max,num=sidelength)
    y_range = np.linspace(start=y_min,stop=y_max,num=sidelength)
    # field will contain all plot locations, encoded in qubits
    field = np.zeros((sidelength,sidelength,model.n_QuBit*model.information_density))
    for i in range(sidelength):
        for j in range(sidelength):
            if n_QuBit*information_density == 2:
                field[i,j,:] = [x_range[i],y_range[j]]
            elif n_QuBit*information_density == 4:
                field[i,j,:] = [x_range[i],y_range[j],x_range[i],y_range[j]]
            elif n_QuBit*information_density == 8:
                field[i,j,:] = [x_range[i],y_range[j],x_range[i],y_range[j],x_range[i],y_range[j],x_range[i],y_range[j]]

for epoch in range(n_epochs):
    print("epoch {} : ".format(epoch),end="")
    model.fit(X_train,Y_train,X_val,Y_val,print_metrics=True,print_gradients=False)

    if show_epochs or save_epochs:  # plotting the decision boundaries for every epoch:
        if save_epochs:
            savepath = dirname + "/epoch{}.pdf".format(epoch)
        else:
            savepath = ""

        plot_predictions(
            model,
            field,
            (X_train,Y_train,"Training set","x"),
            (X_val,Y_val,"Validation Set","."),
            figname="epoch ".format(epoch),
            # figtitle="epoch {} took {:.3f} seconds.".format(epoch,model.epoch_times[-1]),
            closeplot = not show_epochs,
            savepath=savepath
        )

    if model.optimizer == "COBYLA":
        model.new_params()

if write_metrics:   # writing the metrics to a csv file
    # creating the file:
    file = open(dirname + "/metrics.csv","w")
    file.write("Metrics (epoch,Loss_train,Accuracy_train,Precision_train,Recall_train,Loss_val,Accuracy_val,Precision_val,Recall_val,time):")
    for i in range(n_epochs):
        file.write("\n{},".format(i))
        file.write("{:.4f},{:.4f},{:.4f},{:.4f},".format(model.train_metrics["loss"][i],model.train_metrics["accuracy"][i],model.train_metrics["precision"][i],model.train_metrics["recall"][i]) +
                   "{:.4f},{:.4f},{:.4f},{:.4f},".format(model.val_metrics["loss"][i],model.val_metrics["accuracy"][i],model.val_metrics["precision"][i],model.val_metrics["recall"][i]) +
                   "{:.4f}".format(model.epoch_times[i]))
    file.close()

if write_gradients and optimizer != "COBYLA" and implementation != "classical":   # writing the gradients to a csv file
    # creating the file:
    file = open(dirname + "/gradients.csv","w")
    file.write("Gradients (format: 0th component is the epoch, following entries are the gradient components w,b,phi,eta1,eta2):")
    for i in range(n_epochs):
        file.write("\n{}".format(i))
        for comp in model.grad[i]:
            file.write(",{:.5e}".format(comp))
    file.close()

if write_metrics and implementation != "classical":   # writing the parameter history to a csv file
    # creating the file:
    file = open(dirname + "/params.csv","w")
    file.write("parameters (format: 0th component is the epoch, following entries are the parameters w,b,phi,eta1,eta2):")
    for i in range(n_epochs):
        file.write("\n{}".format(i))
        for param in model.params[i]:
            file.write(",{:.5e}".format(param))
    file.close()

if show_results or save_results:    # plotting the metrics for all epochs
    ncols = 2 if model.implementation == "classical" else 3
    fig,ax = plt.subplots(nrows=1,ncols=ncols,figsize=(30,10),num="plot_metrics")
    epoch_axis = np.linspace(start = 1,stop=n_epochs,num=n_epochs)

    # title of the figure
    title = "implementation = " + model.implementation + " | loss function = cross\\_entropy | learning rate = {} | n\\_layers = {}".format(model.learning_rate,model.n_layers) + " | mean epoch time = {:.3f}s".format(np.mean(model.epoch_times))
    if model.implementation == "classical":
        node_structure_string = "({}".format(model.node_structure[0])
        for node in model.node_structure[1:]:
            node_structure_string += ",{}".format(node)
        node_structure_string += ")"
        title += "\nnode\\_structure = "
        title += node_structure_string
    fig.suptitle(title)

    # plotting the metrics
    for metric in ["accuracy","precision","recall"]:
        ax[0].plot(epoch_axis,model.train_metrics[metric],label="train\\_"+metric)
        ax[0].plot(epoch_axis,model.val_metrics[metric],label="val\\_"+metric)
    ax[0].set_title("Metrics")
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel("\\#epoch")
    ax[1].plot(epoch_axis,model.train_metrics["loss"],label="train\\_loss")
    ax[1].plot(epoch_axis,model.val_metrics["loss"],label="val\\_loss")
    ax[1].set_title("Loss")
    ax[1].legend()
    ax[1].grid()
    ax[1].set_xlabel("\\#epoch")
    if model.implementation != "classical" and model.optimizer == "vanillaGD":
        ax[2].plot(epoch_axis+1,[np.sqrt(np.sum(grad**2)) for grad in model.grad],label="$|\\nabla Q|$")
        ax[2].set_title("Norm of the Gradient")
        ax[2].legend()
        ax[2].grid()
        ax[2].set_xlabel("\\#epoch")
    if save_results:
        plt.savefig("./" + dirname + "/metrics.pdf",bbox_inches="tight")
    if not show_results:
        plt.close("plot_metrics")

if show_epochs or show_results:
    plt.show()
plt.close("all")