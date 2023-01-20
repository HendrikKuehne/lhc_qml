import pennylane as qml
from pennylane import numpy as np
from tensorflow import keras
import scipy.optimize as opt
import scipy.linalg as linalg
import copy
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # new axis for colorbars

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "axes.labelsize": 12,
    "xtick.labelsize": 12
})

# ---------------------------------------------------------------------------------------------------------------------------
#                                               miscellaneous
# ---------------------------------------------------------------------------------------------------------------------------

def MatrixScaleUp(matrix,target_size):
    """
    Scaling an (...,N,N)--array to an array with shape (...,target_size,target_size)
    where target_size % matrix.shape[0] == 0.
    """
    old_shape = np.array(matrix.shape)
    assert old_shape[0] != 1
    assert old_shape[-2] == old_shape[-1]
    assert target_size >= old_shape[-1]
    assert old_shape[-1] % 2 == 0
    assert target_size % 2 == 0
    assert target_size % old_shape[-1] == 0

    # creating the new matrix
    new_shape = np.array(matrix.shape)
    new_shape[-1] = target_size
    new_shape[-2] = target_size
    new_matrix = np.ones(shape=new_shape,dtype=complex)

    ratio = int(target_size/old_shape[-1])
    # the old stuff
    # for i in range(old_shape[-1]):
    #     for j in range(old_shape[-1]):
    #         swapped_axis = np.swapaxes(new_matrix[...,ratio*i:ratio*(i+1),ratio*j:ratio*(j+1)],0,-1) * matrix[...,i,j]
    #         new_matrix[...,ratio*i:ratio*(i+1),ratio*j:ratio*(j+1)] = np.swapaxes(swapped_axis,0,-1)

    if len(new_shape) == 2:
        return np.kron(matrix,np.ones(shape=(ratio,ratio),dtype=complex))
    else:
        # the old stuff
        # temp = np.array([np.kron(matrix[i,:,:],np.ones(shape=(ratio,ratio),dtype=complex)) for i in range(new_shape[0])],dtype=complex)
        # return temp
        
        # array for accessing the indices in which the matrices which are to be scaled up are saved:
        it_array = np.zeros(shape=new_shape[0:-2])
        # iterator object:
        it = np.nditer(it_array,flags=["multi_index"])
        # looping over all locations in which the matrices to be scaled up lie:
        for loc in it:
            new_matrix[it.multi_index] = np.kron(matrix[it.multi_index],np.ones(shape=(ratio,ratio)))

        return new_matrix

def plot_predictions(model,field,*datasets,figname="",figtitle="",showplot=False,closeplot=True,savepath=""):
    """
    Plotting the prediction using model.predict() for every entry in field[i,j,...] (for all i,j). It is assumed that field[i,j,0:2] = [x,y] for every point in field.\n
    datasets consists of array_like with len(4) containing points, the respective label, the name of the dataset and the marker used for plotting.\n
    If savepath is given, the plot will be saved under the corresponding path. savepath must contain the file extension.\n
    If showplot == False and closeplot == False, the figure is returned.
    """
    assert model.n_QuBit*model.information_density == field.shape[2]

    fig = plt.figure(num=figname,figsize=(15,15))
    # figure title
    plt.gca().set_title(figtitle)

    # plotting the network prediction:
    pred = model.predict(x=field)
    # pred = np.zeros((field.shape[0],field.shape[1]))
    # for i in range(field.shape[0]):
    #     pred[i,:] = model.predict(x=field[i,:,:])
    field_image = plt.imshow(X=np.transpose(pred),cmap="bwr",origin="lower",extent=(np.amin(field[:,:,0]),np.amax(field[:,:,0]),np.amin(field[:,:,1]),np.amax(field[:,:,1])),alpha=.5)

    # plotting the datasets:
    for dataset in datasets:
        X = dataset[0]
        Y = dataset[1]
        name = dataset[2]
        marker = dataset[3]
        assert X.shape[0] == Y.shape[0]
        
        colors = ["red" if y else "blue" for y in Y]
        if Y.shape[0] > 600:
            plt.scatter(x=X[::int(X.shape[0]/600),0],y=X[::int(X.shape[0]/600),1],c=colors[::int(X.shape[0]/600)],marker=marker,label=name)
        else:
            plt.scatter(x=X[:,0],y=X[:,1],c=colors,marker=marker,label=name)

    plt.legend()
    for i in range(len(datasets)):
        plt.gca().get_legend().legendHandles[i].set_color("black")

    # plotting a colorbar
    divider = make_axes_locatable(plt.gca())
    color_ax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(field_image,orientation="vertical",cax=color_ax)

    if savepath != "":
        plt.savefig("./" + savepath,bbox_inches="tight")
    if showplot:
        plt.show()
    elif closeplot:
        plt.close()
    else:
        return fig

def plot_bloch_sphere(states,labels = np.nan):
    """
    Plots the states given in states on the bloch sphere. states must be a numpy with shape (...,2) where the number of points
    to be plotted, if more than one, is the size of the first axis.\n
    labels is an array containing the label of every data point (1 == signal, 0 == background). When plotting with labels, the
    same colormap is used as in plot_predictions, which is bwr.
    """
    assert states.shape[-1] == 2

    # setting up the figure:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.set_zlim(-1.1,1.1)

    # creating a mesh representing the sphere:
    n = 5
    sphere_mesh_params = np.array([(theta,phi) for theta in np.arange(0,np.pi,np.pi/n) for phi in np.arange(0,2*np.pi,2*np.pi/n)])
    sphere_mesh_points = np.zeros(shape=(sphere_mesh_params.shape[0],3))
    sphere_mesh_points[:,0] = np.sin(sphere_mesh_params[:,0])*np.cos(sphere_mesh_params[:,1])
    sphere_mesh_points[:,1] = np.sin(sphere_mesh_params[:,0])*np.sin(sphere_mesh_params[:,1])
    sphere_mesh_points[:,2] = np.cos(sphere_mesh_params[:,0])

    # plotting
    # ax.scatter(sphere_mesh_points[:,0],sphere_mesh_points[:,1],sphere_mesh_points[:,2],color="gray",alpha=.5,zorder=1)

    if len(states.shape) == 1:
        points_3d = np.zeros(shape=(1,3),dtype=float)
    else:
        points_3d = np.zeros(shape=(states.shape[0],3),dtype=float)

    # extracting the angles on the bloch sphere from the states, element-wise:
    theta = 2*np.arccos(np.abs(states[...,0]))
    phi = (np.angle(states[...,1]) - np.angle(states[...,0])) % (2*np.pi)

    # writing the 3d-points (spherical coordinates):
    points_3d[...,0] = np.sin(theta)*np.cos(phi)
    points_3d[...,1] = np.sin(theta)*np.sin(phi)
    points_3d[...,2] = np.cos(theta)

    # plotting
    if not np.isnan(labels).any():
        assert len(labels) == points_3d.shape[0]
        ax.scatter(points_3d[...,0],points_3d[...,1],points_3d[...,2],c=labels,cmap="bwr",zorder=10)
    else:
        ax.scatter(points_3d[...,0],points_3d[...,1],points_3d[...,2],zorder=10)
    
    plt.show()

# ---------------------------------------------------------------------------------------------------------------------------
#                                               generating datasets
# ---------------------------------------------------------------------------------------------------------------------------

def make_lines(d,l,phi,n_samples=100,noise=1):
    """
    generating a dataset which consists of two parallel lines. Centered on the origin.\n
    d : distance of the lines to the origin\n
    l : length of the lines\n
    phi : angle enclosed by the x-axis and a line perpendicular to both lines to be generated (0 < phi < pi)
    """
    phi = phi%np.pi
    n1 = int(n_samples/2)
    # for the line in the upper half (y > 0)
    x1R = d*np.cos(phi) - l*np.sin(phi)/2
    y1R = d*np.sin(phi) + l*np.cos(phi)/2
    x2R = d*np.cos(phi) + l*np.sin(phi)/2
    y2R = d*np.sin(phi) - l*np.cos(phi)/2
    # for the line in the lower half (y < 0)
    x1L = -d*np.cos(phi) - l*np.sin(phi)/2
    y1L = -d*np.sin(phi) + l*np.cos(phi)/2
    x2L = -d*np.cos(phi) + l*np.sin(phi)/2
    y2L = -d*np.sin(phi) - l*np.cos(phi)/2
    points = np.zeros((n_samples,3))    # third component along second axis is the label
    for i in range(n1):     # line in the upper half - signal
        x = x1R + (x2R - x1R)*i/n1
        y = y1R + (y2R - y1R)*i/n1
        points[i,:] = [x,y,1]
    for i in range(n1,n_samples):   # line in the lower half - background
        x = x1L + (x2L - x1L)*(i - n1)/(n_samples - n1)
        y = y1L + (y2L - y1L)*(i - n1)/(n_samples - n1)
        points[i,:] = [x,y,0]
    
    np.random.shuffle(points)
    points[:,0:2] += np.random.normal(scale=noise,size=(n_samples,2))

    return points[:,0:2],points[:,2]

# ---------------------------------------------------------------------------------------------------------------------------
#                                               defining gates
# ---------------------------------------------------------------------------------------------------------------------------

def GateComposer(*gates,n_wires=0):
    """
    Computes the tensor product of all given gates. When any of the gates provided has multiple channels, the
    resulting gate will have the same number of channels. If there are wires on which no qubit acts, the
    identity gate will be inserted. If n_Qubits is not provided, the number of wires will be set to the last
    wire on which a gate acts (the first wire has index 0).
    """

    # checking if all gates have the correct number of channels:
    n_channels = 1
    for gate in gates:
        if gate.n_channels != 1:
            if n_channels == 1:
                n_channels = gate.n_channels
            else:
                assert n_channels == gate.n_channels, "missmatch in channel number of gate " + gate.name

    # unpacking the wires:
    wires = [gate.wires for gate in gates]
    for wire in wires:
        if not np.isscalar(wire):
            wires = [*wires,*wire]
            wires.remove(wire)
    
    # Checking if all wires are accounted for and, if not, filling up with identities:
    for i in range(max(max(wires)+1,n_wires+1)):
        if i not in wires:
            gates += (Id(wires=i),)
            wires.append(i)
    wires = np.array(wires)
    for i in range(max(wires)+1):
        assert sum(wires == i) == 1, "Two operations are applied to qubit {}".format(i)

    # initialising the complete gate:
    n_QuBits = 0
    tensor_size = 1
    for gate in gates:
        n_QuBits += gate.n_QuBits
        tensor_size *= 2**(gate.n_QuBits)

    if n_channels == 1:
        tensor_gate = np.ones(shape=(tensor_size,tensor_size),dtype=complex)
    else:
        tensor_gate = np.ones(shape=(*n_channels,tensor_size,tensor_size),dtype=complex)

    # writing all gates to the tensor gate:
    for gate in gates:
        if gate.n_QuBits == 1:
            # size of the submatrices to which the gate-matrix will be scaled up
            submatrix_size = int(tensor_size/(2**(gate.wires)))

            for i in range(2**(gate.wires)):
                for j in range(2**(gate.wires)):      # looping over the submatrices
                    # identifying the submatrix (i,j) and writing the entries of the gate to be applied
                    tensor_gate[...,submatrix_size*i:submatrix_size*(i+1),submatrix_size*j:submatrix_size*(j+1)] *= MatrixScaleUp(gate.matrix,submatrix_size)

        elif gate.n_QuBits == 2:
            # size of the outer and inner frame:
            outer_frame_size = int(tensor_size/(2**(gate.wires[0])))     # size of the qubit wires[0]
            inner_frame_size = int(tensor_size/(2**(gate.wires[1])))     # size of the qubit wires[1]

            for i in range(2**(gate.wires[0])):
                for j in range(2**(gate.wires[0])):     # looping over the outer frames
                    outer_frame = tensor_gate[...,outer_frame_size*i:outer_frame_size*(i+1),outer_frame_size*j:outer_frame_size*(j+1)]

                    for k in range(int(outer_frame_size/(2*inner_frame_size))):
                        for l in range(int(outer_frame_size/(2*inner_frame_size))):     # looping over the inner frames
                            # writing the quadrants of the gate matrix to the respective inner frames
                            outer_frame[...,inner_frame_size*k:inner_frame_size*(k+1),inner_frame_size*l:inner_frame_size*(l+1)] *= MatrixScaleUp(gate.matrix[0:2,0:2],target_size=inner_frame_size)
                                # 00-quadrant
                            outer_frame[...,int(outer_frame_size/2) + inner_frame_size*k:int(outer_frame_size/2) + inner_frame_size*(k+1),inner_frame_size*l:inner_frame_size*(l+1)] *= MatrixScaleUp(gate.matrix[2:4,0:2],target_size=inner_frame_size)
                                # 10-quadrant
                            outer_frame[...,inner_frame_size*k:inner_frame_size*(k+1),int(outer_frame_size/2) + inner_frame_size*l:int(outer_frame_size/2) + inner_frame_size*(l+1)] *= MatrixScaleUp(gate.matrix[0:2,2:4],target_size=inner_frame_size)
                                # 01-quadrant
                            outer_frame[...,int(outer_frame_size/2) + inner_frame_size*k:int(outer_frame_size/2) + inner_frame_size*(k+1),int(outer_frame_size/2) + inner_frame_size*l:int(outer_frame_size/2) + inner_frame_size*(l+1)] *= MatrixScaleUp(gate.matrix[2:4,2:4],target_size=inner_frame_size)
                                # 11-quadrant
        
        elif gate.n_QuBits > 2:
            assert False, "trying to apply gate which applies to more than 2 QuBits"

    return tensor_gate

# Every class representing a gate should have four attributes:
#       .n_QuBits : The number of QuBits it takes
#       .wires : The specific QuBit(s) it acts on. Should return a list if n_QuBits > 1.
#       .matrix : matrix representation of the gate in the computational basis as numpy-array.
#       .n_channels : number of channels
# When a class which takes a parameter receives an array with shape n, the attribute matrix has shape (n,2,2),
# meaning for every parameter (every "channel"), an array is created. The number of parameters s, in this case,
# determines the attribute n_channels.

class Id:
    """
    The identity gate acting on the QuBits specified in wire. The QuBits given in wire must be consecutive.
    """
    def __init__(self,wires):
        if np.isscalar(wires):
            self.n_QuBits = 1
        else:
            self.n_QuBits = len(wires)

        self.wires = wires
        self.matrix = np.diag(np.ones(2**self.n_QuBits))
        self.name = "Id"
        self.n_channels = 1

class CNOT:
    """
    The CNOT-gate. the control-wire serves as control, the invert-wire will be inverted.
    Acts as an X-gate if control == invert.
    """
    def __init__(self,control,invert):
        self.name = "CNOT"
        self.n_channels = 1

        if control < invert:
            self.n_QuBits = 2
            self.wires = [control,invert]
            self.matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)
        elif control > invert:
            self.n_QuBits = 2
            self.wires = [invert,control]
            self.matrix = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]],dtype=complex)
        elif control == invert:
            self.n_QuBits = 1
            self.wires = control
            self.matrix = np.array([[0,1],[1,0]],dtype=complex)

class SWAP:
    """The SWAP-Gate. Acts as identity if the same wire is provided two times."""
    def __init__(self,wires):
        self.name = "SWAP"
        self.n_channels = 1
        self.n_QuBits = 1

        if np.isscalar(wires):
            self.n_QuBits = 1
            self.wires = wires
            self.matrix = np.array([[1,0],[0,1]],dtype=complex)
        else:
            assert len(wires) == 2
            self.n_QuBits = 2
            self.wires = np.array(wires)
            self.matrix = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=complex)

class PauliX:
    def __init__(self,wires):
        self.name = "PauliX"
        self.wires = wires
        self.n_QuBits = 1
        self.n_channels = 1
        self.matrix = np.array([[0,1],[1,0]],dtype=complex)

class PauliY:
    def __init__(self,wires):
        self.name = "PauliY"
        self.n_channels = 1
        self.wires = wires
        self.n_QuBits = 1
        self.matrix = np.array([[0,-1.j],[1.j,0]],dtype=complex)

class PauliZ:
    def __init__(self,wires):
        self.name = "PauliZ"
        self.n_channels = 1
        self.wires = wires
        self.n_QuBits = 1
        self.matrix = np.array([[1,0],[0,-1]],dtype=complex)

class RX:
    """
    RX-gate. If, at initialisation, a numpy-array with arbitrary shape is given for theta, all elements in the last dimension of theta will be replaced with the corresponding gate, thus, (2,2) will be appended to .matrix.shape.
    """
    def __init__(self,wires,theta):
        self.name = "RX"
        self.wires = wires
        self.n_QuBits = 1

        # checking if we are supplied with one or multiple parameters:
        if hasattr(theta,"shape"):
            if len(theta.shape) == 0:
                scalar = True
            else:
                scalar = False

        if np.isscalar(theta) or scalar:
            self.matrix = np.array([[np.cos(theta/2),-1.j*np.sin(theta/2)],[-1.j*np.sin(theta/2),np.cos(theta/2)]],dtype=complex)
            self.n_channels = 1
        else:
            theta = np.array(theta)
            self.matrix = np.zeros(shape=(*theta.shape,2,2),dtype=complex)
            self.matrix[...,0,0] = np.cos(theta/2)
            self.matrix[...,0,1] = -1.j*np.sin(theta/2)
            self.matrix[...,1,0] = -1.j*np.sin(theta/2)
            self.matrix[...,1,1] = np.cos(theta/2)

            self.n_channels = theta.shape

class RY:
    """
    RY-gate. If, at initialisation, a numpy-array with arbitrary shape is given for theta, all elements in the last dimension of theta will be replaced with the corresponding gate, thus, (2,2) will be appended to .matrix.shape.
    """
    def __init__(self,wires,theta):
        self.name = "RY"
        self.wires = wires
        self.n_QuBits = 1

        # checking if we are supplied with one or multiple parameters:
        if hasattr(theta,"shape"):
            if len(theta.shape) == 0:
                scalar = True
            else:
                scalar = False

        if np.isscalar(theta) or scalar:
            self.matrix = np.array([[np.cos(theta/2),-np.sin(theta/2)],[np.sin(theta/2),np.cos(theta/2)]],dtype=complex)
            self.n_channels = 1
        else:
            theta = np.array(theta)
            self.matrix = np.zeros(shape=(*theta.shape,2,2),dtype=complex)
            self.matrix[...,0,0] = np.cos(theta/2)
            self.matrix[...,0,1] = -np.sin(theta/2)
            self.matrix[...,1,0] = np.sin(theta/2)
            self.matrix[...,1,1] = np.cos(theta/2)

            self.n_channels = theta.shape

class RZ:
    """
    RZ-gate. If, at initialisation, a numpy-array with arbitrary shape is given for theta, all elements in the last dimension of theta will be replaced with the corresponding gate, thus, (2,2) will be appended to .matrix.shape.
    """
    def __init__(self,wires,theta):
        self.name = "RZ"
        self.wires = wires
        self.n_QuBits = 1

        # checking if we are supplied with one or multiple parameters:
        if hasattr(theta,"shape"):
            if len(theta.shape) == 0:
                scalar = True
            else:
                scalar = False

        if np.isscalar(theta) or scalar:
            self.matrix = np.array([[np.exp(-1.j*theta/2),0],[0,np.exp(1.j*theta/2)]],dtype=complex)
            self.n_channels = 1
        else:
            theta = np.array(theta)
            self.matrix = np.zeros(shape=(*theta.shape,2,2),dtype=complex)
            self.matrix[...,0,0] = np.exp(-1.j*theta/2)
            self.matrix[...,1,1] = np.exp(1.j*theta/2)

            self.n_channels = theta.shape

class Rot:
    """
    Rot-gate. If, at initialisation, a numpy-array with arbitrary shape is given for theta, all elements in the last dimension of theta will be replaced with the corresponding gate, thus, (2,2) will be appended to .matrix.shape.
    """
    def __init__(self,wires,params):
        params = np.array(params)
        assert params.shape[-1] == 3

        self.name = "Rot"
        self.wires = wires
        self.n_QuBits = 1

        # checking if we are supplied with one or multiple parameter sets:
        if len(params.shape) == 1:
            self.n_channels = 1
        else:
            self.n_channels = params.shape[0:-1]

        phi = params[...,0]
        theta = params[...,1]
        omega = params[...,2]

        if self.n_channels == 1:
            self.matrix = np.array([[np.exp(-1.j*(phi+omega)/2)*np.cos(theta/2),-np.exp(1.j*(phi-omega)/2)*np.sin(theta/2)],[np.exp(-1.j*(phi-omega)/2)*np.sin(theta/2),np.exp(1.j*(phi+omega)/2)*np.cos(theta/2)]],dtype=complex)
        else:
            self.matrix = np.zeros(shape=(*self.n_channels,2,2),dtype=complex)
            self.matrix[...,0,0] =  np.exp(-1.j*(phi+omega)/2)*np.cos(theta/2)
            self.matrix[...,0,1] = -np.exp( 1.j*(phi-omega)/2)*np.sin(theta/2)
            self.matrix[...,1,0] =  np.exp(-1.j*(phi-omega)/2)*np.sin(theta/2)
            self.matrix[...,1,1] =  np.exp( 1.j*(phi+omega)/2)*np.cos(theta/2)

class H:
    """
    Hadamard gate.
    """
    def __init__(self,wires):
        self.name = "Hadamard"
        self.wires = wires
        self.n_QuBits = 1
        self.n_channels = 1
        self.matrix = np.array([[1,1],[1,-1]],dtype=complex)/np.sqrt(2)

class Ph:
    """
    Phase shift gate. Apllies a phase shift to state |1>. If, at initialisation, a numpy-array with arbitrary shape is given for theta,
    all elements in the last dimension of theta will be replaced with the corresponding gate, thus, (2,2) will be appended to .matrix.shape.
    """
    def __init__(self,wires,theta):
        self.name = "Phase shift"
        self.wires = wires
        self.n_QuBits = 1

        # checking if we are supplied with one or multiple parameters:
        if hasattr(theta,"shape"):
            if len(theta.shape) == 0:
                scalar = True
            else:
                scalar = False

        if np.isscalar(theta) or scalar:
            self.matrix = np.array([[1,0],[0,np.exp(theta*1.j)]],dtype=complex)
            self.n_channels = 1
        else:
            theta = np.array(theta)
            self.matrix = np.zeros(shape=(*theta.shape,2,2),dtype=complex)
            self.matrix[...,0,0] = 1
            self.matrix[...,1,1] = np.exp(theta*1.j)

            self.n_channels = theta.shape

class IsingTime:
    """
    Time evolution under the hamiltonian of the fully connected, transverse Ising model with random coupling strengths. Default time is t = 10.
    """
    def __init__(self,n_QuBit,t=10):
        self.name = "Ising Time Evolution"
        self.n_QuBits = n_QuBit
        self.n_channels = 1
        
        if n_QuBit == 1:
            self.wires = 0
        else:
            self.wires = np.linspace(start=0,stop=n_QuBit-1,num=n_QuBit,dtype=int)

        # generating the hamiltonian
        H = np.zeros(shape=(2**(n_QuBit),2**(n_QuBit)),dtype=complex)
        generator = np.random.default_rng(42)
        # effect of external field
        for i in range(n_QuBit):
            H += generator.uniform(low=-1,high=1)*GateComposer(PauliX(wires=i),n_wires=n_QuBit-1)
        # effect of coupling:
        for i in range(n_QuBit-1):
            H += generator.uniform(low=-1,high=1)*GateComposer(PauliZ(wires=i),PauliZ(wires=i+1),n_wires=n_QuBit-1)
        
        # matrix exponential:
        self.matrix = linalg.expm(-H*t*1.j)

class W:
    """
    W-gate, used for wavefunction-encoding. If, at initialisation, numpy-arrays with arbitrary shapes are given for x1,x2, all elements in the last dimension of x1 will be replaced with the corresponding gate, i.e., (2,2) will be appended to x1.shape.
    """
    def __init__(self,wires,x1,x2):
        self.name = "W"
        self.wires = wires
        self.n_QuBits = 1

        # checking if we are supplied with one or multiple parameters:
        if hasattr(x1,"shape") or hasattr(x2,"shape"):
            assert hasattr(x1,"shape") and hasattr(x2,"shape")
            assert x1.shape == x1.shape
            scalar = False
        else:
            assert np.isscalar(x1) and np.isscalar(x2)
            scalar = True
        
        # defining the matrix
        if scalar:
            denom = np.sqrt(x1**2 + x1**2)
            self.matrix = np.array([[x1/denom,0],[x2/denom,0]],dtype=complex)
            self.n_channels = 1
        else:
            denom = np.sqrt(x1**2 + x2**2)
            self.matrix = np.zeros(shape=(*x1.shape,2,2),dtype=complex)
            self.matrix[...,0,0] = x1/denom
            self.matrix[...,0,1] = 0
            self.matrix[...,1,0] = x2/denom
            self.matrix[...,1,1] = 0

            self.n_channels = x1.shape

# ---------------------------------------------------------------------------------------------------------------------------
#                                               network infrastructure
# ---------------------------------------------------------------------------------------------------------------------------

def sigmoid(x,a=2):
    """sigmoid function"""
    return 1/(1 + np.exp(-a*x))

def grad_sigmoid(x,a=2):
    """Derivative of the sigmoid function"""
    return a*np.exp(-a*x)/(1+np.exp(-a*x))**2

def compile_network(node_structure,n_features,hidden_activation_function = "relu",output_activation_function = "sigmoid",learning_rate=.05):
    """
    Creating a classical neural network\n
    node_structure : array containing the number of nodes per layer where the length of the array corresponds to the number of layers
    """

    classical_network = keras.models.Sequential()
    classical_network.add(keras.layers.InputLayer(input_shape=(n_features)))
    for nodes in node_structure:
        classical_network.add(keras.layers.Dense(nodes,activation=hidden_activation_function))
    classical_network.add(keras.layers.Dense(1,activation=output_activation_function))

    # classical_network.summary()

    opt = keras.optimizers.SGD(learning_rate=learning_rate)

    classical_network.compile(
        loss = "binary_crossentropy",
        optimizer = opt,
        metrics = ["accuracy","Precision","Recall"]
    )

    return classical_network

# ---------------------------------------------------------------------------------------------------------------------------
#                                               class making the predictions
# ---------------------------------------------------------------------------------------------------------------------------

class predictor:
    """
    Class which evaluates a (quantum) classifier to make predictions after having been trained. Six implementations are available:\n
        scratch: A selfmade quantum circuit using numpy.\n
        pennylane: A quantum circuit implementation using pennylane.\n
        classical: A classical neural network.\n
        prerot: The quantum circuit from "scratch" except that beforehand, the datapoints may be rotated (assumes cartesian coordinates as input).\n
        pretrans: The quantum circuit from "scratch" except that beforehand, the datapoints may be translated (assumes cartesian coordinates as input).\n
        pregalilei: The quantum circuit from "scratch" with a rotation followed by a translation of the datapoints beforehand (as in "prerot" and "pretrans").\n
    information_density : Number of input variables per QuBit.\n
    
    Because the gradient of the model with respect to x is calculated using the parameter shift rule, no inverse trigonometric
    functions may be used for preparation when using the implementation prerot! This is beceause when using the parameter shift
    rule, the normalisation is not necessarily adapted to normalizing values in the range that the parameter shift rule spans.\n
    
    When preparation parameters correspond to cartesian coordinates and n_QuBit > 2, ist is assumed that x is given in consecutive
    pairs of the x1- and x2-ccordinates, eg x = [x1,x2,x1,x2,...]. n_QuBit*information_density must thus be divisible by 2.
    """
    def new_params(self):
        """
        Initializing with new random parameters.
        """
        # rotation angles for rotation gates
        self.w = np.random.uniform(low=-2*np.pi,high=2*np.pi,size=3*self.n_QuBit*self.n_layers)
        # bias added to the PauliZ expectation value - currently disabled
        self.b = 0
        # rotation angle for classical preprocessing
        self.phi = np.random.uniform(low=0,high=2*np.pi) if (self.implementation in ("prerot","pregalilei")) else 0
        # translations of cartesian coordinates in pretrans or pregalilei
        self.eta1 = 0
        self.eta2 = 0
    
    def __init__(self,implementation,n_QuBit,learning_rate=1,node_structure=[],n_layers=4,optimizer="vanillaGD",information_density=1):
        # security check
        assert implementation in ["scratch","pennylane_default","pennylane_lightning","classical","prerot","pretrans","pregalilei"], "wrong value for variable implementation in initialisation of predictor"
        assert optimizer in ["vanillaGD","COBYLA"]
        
        if implementation in ("prerot","pretrans","pregalilei"):
            assert ((n_QuBit*information_density) % 2) == 0

        # hyperparameters
        self.learning_rate = learning_rate
        self.implementation = implementation
        self.n_QuBit = n_QuBit
        self.n_layers = n_layers
        self.optimizer = optimizer

        # metrics
        self.train_metrics = {"loss":[],"accuracy":[],"precision":[],"recall":[]}
        self.val_metrics = {"loss":[],"accuracy":[],"precision":[],"recall":[]}
        self.epoch_times = []
        self.grad = []
        self.params = []

        # initialising the parameters randomly:
        self.new_params()
        
        # network infrastructure
        self.norm_defined = False
        self.information_density = information_density

        if implementation == "classical":
            self.n_layers = len(node_structure)
            self.node_structure = node_structure
            # creating a neural network
            self.classical_network = compile_network(node_structure=node_structure,n_features=n_QuBit*information_density,learning_rate=learning_rate)

        elif implementation in ("scratch","prerot","pretrans","pregalilei"):
            # defining the entanglement layer:
            if n_QuBit > 1:
                # entanglement_CNOT = []
                # for i in range(n_QuBit):
                #     for j in range(n_QuBit):
                #         if i != j:
                #             entanglement_CNOT += [GateComposer(CNOT(control=i,invert=j),n_wires=n_QuBit-1),]
                # # reversing because the order in the list does not match the order in which the gates are applied
                # entanglement_CNOT.reverse()
                # # Multiplying all CNOT-gates beforehand so that less matrix multplication has to be carried out during predictions:
                # self.entanglement_gate = np.linalg.multi_dot(entanglement_CNOT)
                self.entanglement_gate = IsingTime(n_QuBit=self.n_QuBit).matrix
            else:
                self.entanglement_gate = Id(wires=0)
            
        elif implementation in ("pennylane_default","pennylane_lightning"):
            if implementation == "pennylane_default":
                self.qmldev = qml.device(name="default.qubit",wires=n_QuBit)
            else:
                self.qmldev = qml.device(name="lightning.qubit",wires=n_QuBit)
            
            # creating the hamiltonian used for entanglement:
            coeffs = np.random.uniform(low=-1,high=1,size=2*n_QuBit-1)
            observables = [qml.PauliX(wires=i) for i in range(n_QuBit)] + [qml.PauliZ(wires=i)@qml.PauliZ(wires=i+1) for i in range(n_QuBit-1)]
            hamiltonian = qml.Hamiltonian(coeffs=coeffs,observables=observables)

            @qml.qnode(self.qmldev)
            def quantum_circuit(x,w,b):
                # state preparation
                for i in range(n_QuBit):
                    # qml.Hadamard(wires=i)
                    # qml.PhaseShift(phi=np.arccos(x[...,i]),wires=i)
                    qml.RY(phi=x[...,i],wires=i)
                    # qml.RZ(phi=np.arccos(x[...,i]**2),wires=i)

                for h in range(self.n_layers):      # looping over all layers
                    # applying rotation gates:
                    for i in range(n_QuBit):
                        qml.Rot(phi=w[3*self.n_QuBit*h+3*i],theta=w[3*self.n_QuBit*h+3*i+1],omega=w[3*self.n_QuBit*h+3*i+2],wires=i)
                    
                    # applying entanglement gates:
                    for i in range(n_QuBit):
                        for j in range(n_QuBit):
                            if i != j:
                                qml.CNOT(wires=(i,j))
                    # qml.templates.ApproxTimeEvolution(hamiltonian=hamiltonian,time=10,n=100)

                return qml.expval(qml.PauliZ(wires=0))
                # return qml.state()
            
            self.quantum_circuit = quantum_circuit

    def normalize(self,x=np.nan,min=np.nan,max=np.nan,min_norm = 0,max_norm=2*np.pi):
        """
        When first called, the parameters which determine the normalisation will be set. For every QuBit, the input will be normalised to a range of [min_norm,max_norm]. Feature i in the input data will behave as follows under normalisation: min[i] -> min_norm and max[i] -> max_norm.\n
        On subsequent calls, the normalised data will be returned.\n
        If no parameters have been set, x will be returned without alterations.
        """
        if (not self.norm_defined) and (not np.isnan(min).all()) and (not np.isnan(max).all()):
            # defining parameters
            assert len(min) == self.n_QuBit*self.information_density
            assert len(max) == self.n_QuBit*self.information_density
            assert min_norm < max_norm

            self.x_min = np.array(min)
            self.x_max = np.array(max)
            self.min_norm = min_norm
            self.max_norm = max_norm

            self.norm_defined = True
        elif (not self.norm_defined):
            # parameters are not defined
            return copy.deepcopy(x)
        elif self.norm_defined:
            # parameters are defined
            new_x = np.zeros(shape=x.shape)
            for i in range(self.n_QuBit*self.information_density):
                new_x[...,i] = self.min_norm + (x[...,i] - self.x_min[i])*(self.max_norm - self.min_norm)/(self.x_max[i] - self.x_min[i])
            return new_x

    def predict(self,w=np.nan,b=np.nan,phi=np.nan,eta1=np.nan,eta2=np.nan,x=np.nan,return_expval=False):
        """
        Making predictions for the input x. Returns array if x is an array. If a parameter is not given, the internal
        value will be used. Normalisation is done if the corresponding parameters have been set using
        predictor.normalize.\n
        If return_expval == True, the PauliZ expectation value is returned instead of the value of the sigmoid function.
        """
        # normalisation if the corresponding parameters have been defined:
        x = self.normalize(x)

        if self.implementation != "classical":    # checking whether parameters are provided or the internal values are used
            if np.isnan(w).any():
                w = self.w
            if np.isnan(b):
                b = self.b
            if np.isnan(phi):
                phi = self.phi
            if np.isnan(eta1):
                eta1 = self.eta1
            if np.isnan(eta2):
                eta2 = self.eta2
            assert len(w) == 3*self.n_QuBit*self.n_layers, "provided the wrong number of parameters"

        assert x.shape[-1] == self.n_QuBit*self.information_density, "Wrong shape for input variable x"

        if self.implementation == "classical":
            # reshaping
            old_shape = x.shape
            x = x.reshape([-1,self.n_QuBit*self.information_density])
            # making a prediction
            return self.classical_network.predict(x,verbose=0).flatten().reshape(old_shape[:-1])

        # classical preprocessing
        if self.implementation in ("prerot","pregalilei"):      # rotation
            rot_mat = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
            # x_old = x
            for i in range(int(x.shape[-1]/2)):
                x_mean = np.array([np.mean(x[...,2*i]),np.mean(x[...,2*i+1])])
                x[...,2*i:2*i+2] = x_mean + np.einsum("ij,...j->...i",rot_mat,x[...,2*i:2*i+2]) - np.einsum("ij,j->i",rot_mat,x_mean)

        if self.implementation in ("pretrans","pregalilei"):    # translation
            for i in range(int(x.shape[-1]/2)):
                x_mean = np.array([np.mean(x[...,2*i]),np.mean(x[...,2*i+1])])
                x[...,2*i:2*i+2] += np.array([eta1,eta2])

        # quantum circuits
        if self.implementation in ("scratch","prerot","pretrans","pregalilei"):
            if len(x.shape) == 1:
                state = np.zeros(2**(self.n_QuBit),dtype=complex)
            else:
                state = np.zeros(shape=(*x.shape[0:-1],2**(self.n_QuBit)))
            state[...,0] = 1
            # I thought about multiplying all gates together with np.linalg.multi_dot before applying them to the state but the rotation gates
            # may actually be stacks of matrices and multi_dot does not seem to be able to handle that - so I won't

            # preparation gates
            # prep_gates_1 = (RY(wires=i,theta=np.arcsin(x[...,i])) for i in range(self.n_QuBit))
            # prep_gate_1 = GateComposer(*prep_gates_1)
            # prep_gates_2 = (RY(wires=i,theta=np.arccos(x[...,i]**2)) for i in range(self.n_QuBit))
            # prep_gate_2 = GateComposer(*prep_gates_2)
            # prep_gates_mixed = (
            #     RY(wires=0,theta=x[...,0]),
            #     RY(wires=1,theta=x[...,1]),
            #     # RY(wires=2,theta=x[...,2]),
            #     # RY(wires=3,theta=x[...,3]),
            #     RY(wires=2,theta=np.arccos(x[...,2])),
            #     RY(wires=3,theta=np.arccos(x[...,3])),
            #     # RY(wires=6,theta=np.arccos(x[...,6])),
            #     # RY(wires=7,theta=np.arccos(x[...,7]))
            # )
            prep_gates_RY = (RY(wires=i,theta=x[...,i]) for i in range(self.n_QuBit))
            prep_gates_Ph = (Ph(wires=i,theta=x[...,i+1]) for i in range(self.n_QuBit))
            prep_gate_RY = GateComposer(*prep_gates_RY)
            prep_gate_Ph = GateComposer(*prep_gates_Ph)
            # prep_gates_W = (W(wires=i,x1=x[...,i],x2=x[...,i+1]) for i in range(self.n_QuBit))
            # prep_gate_W = GateComposer(*prep_gates_W)

            # preparation layer
            state = np.einsum("...jk,...k->...j",prep_gate_RY,state)
            state = np.einsum("...jk,...k->...j",prep_gate_Ph,state)

            for h in range(self.n_layers):
                # applying rotation gates:
                rot_gates_single = (Rot(wires=i,params=w[3*self.n_QuBit*h+3*i:3*self.n_QuBit*h+3*(i+1)]) for i in range(self.n_QuBit))
                rot_gates = GateComposer(*rot_gates_single)
                state = np.einsum("...jk,...k->...j",rot_gates,state)
                # applying the entanglement gate:
                state = np.einsum("...jk,...k->...j",self.entanglement_gate,state)

            # computing the PauliZ expectation value:
            PauliZ_gate = GateComposer(PauliZ(wires=0),n_wires=self.n_QuBit-1)
            PauliZ_state = np.einsum("...jk,...k->...j",PauliZ_gate,state)
            expval = np.einsum("...j,...j->...",np.conjugate(state),PauliZ_state)

            assert (np.imag(expval) == 0).all(), "complex expectation value in predict()"

            if return_expval:
                return expval.real.astype(float)

            return sigmoid(expval.real.astype(float))

        elif self.implementation in ("pennylane_default","pennylane_lightning"):
            expval = self.quantum_circuit(x,w,b)
            
            if return_expval:
                return expval

            return sigmoid(expval)

    def grad_predictor(self,x,i):
        """
        Derivative of f with respect to the i-th parameter using the parameter shift rule. Takes an array with shape
        (...,n_QuBit*information_density) (the raw data points) and returns an array with shape (...) where n is the number
        of data points. In a more concise way: For every data point, the gradient wiht respect to the i-th parameter is
        calculated.\n

        i : parameter with respect to which the derivative is taken:\n
            i <= 3*n_QuBit*n_layers for the angles of rotation\n
            i == 3*self.n_QuBit*self.n_layers + 1 for the bias b\n
            i == 3*self.n_QuBit*self.n_layers + 2 respectively i == 3*self.n_QuBit*self.n_layers + 3 for the preparation parameters (cartesian x- respectively y-ccordinate))
        """
        r = 1/2
        s = np.pi/(4*r)

        if i < 3*self.n_QuBit*self.n_layers:        # rotation gate angles
            # array used for shifting the parameters in calculation of the gradient
            dw = np.zeros(3*self.n_QuBit*self.n_layers)
            dw[i] = s
            return r*(self.predict(w=self.w+dw,x=x,return_expval=True) - self.predict(w=self.w-dw,x=x,return_expval=True))

        elif i == 3*self.n_QuBit*self.n_layers:     # bias
            return 1

        elif i == 3*self.n_QuBit*self.n_layers + 1:     # cartesian coordinate x1
            # array used for shifting the parameters in calculation of the gradient
            dx1 = np.zeros(self.n_QuBit*self.information_density)
            for i in range(int((self.n_QuBit*self.information_density)/2)):
                dx1[2*i] = s
            return r*(self.predict(x=x+dx1,return_expval=True) - self.predict(x=x-dx1,return_expval=True))
        
        elif i == 3*self.n_QuBit*self.n_layers + 2:     # cartesian coordinate x2
            # array used for shifting the parameters in calculation of the gradient
            dx2 = np.zeros(self.n_QuBit*self.information_density)
            for i in range(int((self.n_QuBit*self.information_density)/2)):
                dx2[2*i] = s
            return r*(self.predict(x=x+dx2,return_expval=True) - self.predict(x=x-dx2,return_expval=True))

    def parameter_update(self,X,x,y,print_gradients=False):
        """
        Updates the parameters of the network and appends the magnitude of the gradient to predictor.grad_norm\n
        X : Raw data points\n
        x : predictions for all data pairs (as computed by predictor.predict(x,w,b)) (included to speed up computation)\n
        y : ground truth values
        """
        assert len(x) == len(y), "Different number of predictions and ground truths provided in parameter_update()"

        w_new = copy.deepcopy(self.w)
        b_new = copy.deepcopy(self.b)
        phi_new = copy.deepcopy(self.phi)
        eta1_new = copy.deepcopy(self.eta1)
        eta2_new = copy.deepcopy(self.eta2)
        
        grad = np.zeros(3*self.n_QuBit*self.n_layers + 4)

        if print_gradients: print("Gradients:")

        # updating the angles in w
        for i in range(len(self.w)):
            grad[i] = (-1)*np.sum((y/x - (1-y)/(1-x))*grad_sigmoid(x)*self.grad_predictor(X,i))/(np.log(2)*len(y))
            w_new[i] -= self.learning_rate*grad[i]

            if print_gradients: print("    grad_{:2} = {:10.3e}".format(i,grad[i]))

        # # updating the bias b
        # grad[-4] = -r*np.sum(y*grad_sigmoid(x)/x)/np.log(2)
        # # grad[-4] = -r*sum(y/x)/np.log(2)
        # b_new -= learning_rate*grad[-2]

        if self.implementation in ("prerot","pregalilei"):          # updating the classical preprocessing angle phi
            grad[-3] = np.sum(
                                (y/x - (1-y)/(1-x))
                               *grad_sigmoid(x)
                               *(
                                    self.grad_predictor(x=X,i=3*self.n_QuBit*self.n_layers + 1)*(-X[:,0]*np.sin(self.phi) + X[:,1]*np.cos(self.phi))
                                   +self.grad_predictor(x=X,i=3*self.n_QuBit*self.n_layers + 2)*(-X[:,0]*np.cos(self.phi) - X[:,1]*np.sin(self.phi))
                                )
                             )/(np.log(2)*len(y))
            phi_new -= self.learning_rate*grad[-3]

        elif self.implementation in ("pretrans","pregalilei"):      # updating the classical preprocessing translations eta1, eta2
            grad[-2] = np.sum(
                                (y/x - (1-y)/(1-x))
                               *grad_sigmoid(x)
                               *self.grad_predictor(x=X,i=3*self.n_QuBit*self.n_layers + 1)
                             )/(np.log(2)*len(y))
            eta1_new -= self.learning_rate*grad[-2]

            grad[-1] = np.sum(
                                (y/x - (1-y)/(1-x))
                               *grad_sigmoid(x)
                               *self.grad_predictor(x=X,i=3*self.n_QuBit*self.n_layers + 2)
                             )/(np.log(2)*len(y))
            eta2_new -= self.learning_rate*grad[-1]

        self.w = w_new
        self.b = b_new
        self.phi = phi_new
        self.eta1 = eta1_new
        self.eta2 = eta2_new
        
        self.grad.append(grad)

    def fit(self,X_train,Y_train,X_val,Y_val,print_metrics=False,return_prediction=False,print_gradients=False):
        """
        Training using X_train,Y_train as training data and X_val,Y_val as validation data.
        The metrics for the epoch are appended to predictor.train_metrics and predictor.epoch_times.\n
        If predictor.optimizer == "vanillaGD" or self.implementation == "classical", one epoch will be performed.\n
        If predictor.optimizer == "COBYLA", the loss function will be minimized as far as possible. Newly generated random parameters will be used as initial guess.\n
        """
        # measuring time
        t0 = time.time()
        
        if self.implementation == "classical":  # updating the parameters:
            # normalization:
            X_train = self.normalize(X_train)
            X_val = self.normalize(X_val)
            # reshaping:
            X_train = X_train.reshape([-1,self.n_QuBit*self.information_density])
            Y_train = Y_train.reshape([-1])
            X_val = X_val.reshape([-1,self.n_QuBit*self.information_density])
            Y_val = Y_val.reshape([-1])
            # training:
            self.classical_network.fit(X_train,Y_train,epochs=1,validation_data=(X_val,Y_val))#),verbose=0)

        elif self.optimizer == "vanillaGD":     # updating the parameters:
            # making a prediction
            x_val = self.predict(x=X_val)
            x_train = self.predict(x=X_train)

            self.parameter_update(X_train,x_train,Y_train,print_gradients=print_gradients)

        elif self.optimizer == "COBYLA":        # Optimizing:
            # function which will be minimized:
            def min_func(param_vec,x=X_train):
                pred = self.predict(w=param_vec[0:-4],b=param_vec[-4],phi=param_vec[-3],eta1=param_vec[-2],eta2=param_vec[-1],x=X_train)
                return Loss(pred,Y_train)
            
            # initial values:
            x0 = np.random.uniform(low=-2*np.pi,high=2*np.pi,size=3*self.n_QuBit*self.n_layers+4)
                # rotation angles stored in x0[0:-4]
            x0[-4] = 0
                # bias stored in x0[-4]
            x0[-3] = np.random.uniform(low=0,high=2*np.pi) if (self.implementation in ("prerot","pregalilei")) else 0
                # phi stored in x0[-3]
            x0[-2] = 0
            x0[-1] = 0
                # translations eta1, eta2 stored in x0[-2], x0[-1]
            
            # minimization:
            result = opt.minimize(fun=min_func,x0=x0,method="COBYLA",options={"disp":True})

            self.w = result.x[0:-4]
            self.b = result.x[-4]
            self.phi = result.x[-3]
            self.eta1 = result.x[-2]
            self.eta2 = result.x[-1]
        
        if self.implementation != "classical":
            self.params.append(np.array([*self.w,self.b,self.phi,self.eta1,self.eta2]))

        # measuring time
        t1 = time.time()

        # making a prediction
        x_val = self.predict(x=X_val)
        x_train = self.predict(x=X_train)

        # saving the metrics
        self.train_metrics["loss"].append(Loss(x_train,Y_train))
        self.train_metrics["accuracy"].append(accuracy(x_train,Y_train))
        self.train_metrics["precision"].append(precision(x_train,Y_train))
        self.train_metrics["recall"].append(recall(x_train,Y_train))

        self.val_metrics["loss"].append(Loss(x_val,Y_val))
        self.val_metrics["accuracy"].append(accuracy(x_val,Y_val))
        self.val_metrics["precision"].append(precision(x_val,Y_val))
        self.val_metrics["recall"].append(recall(x_val,Y_val))

        self.epoch_times.append(t1-t0)

        if print_metrics:
                print("Loss = {:6.4f}   Accuracy = {:6.4f}   Precision = {:6.4f}   Recall = {:6.4f}".format(self.train_metrics["loss"][-1],self.train_metrics["accuracy"][-1],self.train_metrics["precision"][-1],self.train_metrics["recall"][-1])+
                      "   Loss_val = {:6.4f}   Accuracy_val = {:6.4f}   Precision_val = {:6.4f}   Recall_val = {:6.4f}".format(self.val_metrics["loss"][-1],self.val_metrics["accuracy"][-1],self.val_metrics["precision"][-1],self.val_metrics["recall"][-1]),end="")
                print("   t = {:6.4f}s".format(self.epoch_times[-1]))

# ---------------------------------------------------------------------------------------------------------------------------
#                                               metrics
# ---------------------------------------------------------------------------------------------------------------------------

def Loss(x,y,loss_function="cross-entropy"):
    """
    x : predictions by the network\n
    y : ground truth values\n
    loss_function : "l2-norm" or "cross-entropy"
    """
    if not x.shape == y.shape:
        raise Exception("Different number of predictions and ground truths provided in L()")

    if loss_function == "l2-norm":
        return np.sum((y - x)**2)/np.prod(y.shape)
    elif loss_function == "cross-entropy":
        return (-1)*np.sum(y*np.log2(x) + (1 - y)*np.log2(1-x))/np.prod(y.shape)

def accuracy(x,y):
    """
    x : predictions by the network\n
    y : ground truth values
    """
    if not x.shape == y.shape:
        raise Exception("Different number of predictions and ground truths provided in accuracy()")

    x_pred = (x + .5).astype(int)    # converting the confidence of the model into a prediction
    truth = (x_pred == y)

    return np.sum(truth)/np.prod(x_pred.shape)

def precision(x,y):
    """
    x : predictions by the network\n
    y : ground truth values
    """
    # precision = TP/(TP + FP) - out of all positive predictions, how many where true?

    x_pred = (x + .5).astype(int)    # converting the confidence of the model into a prediction
    TP = np.sum(np.logical_and((x_pred==y),y))     # using y+1 since -1 also gets converted to True
    FP = np.sum(np.logical_and((x_pred!=y),x_pred))

    if TP != 0:
        return TP/(TP + FP)
    else:
        return 0

def recall(x,y):
    """
    x : predictions by the network\n
    y : ground truth values
    """
    # recall = TP/(TP + FN) - out of all positive instances, how many were correctly identified?

    x_pred = (x + .5).astype(int)    # converting the confidence of the model into a prediction
    TP = np.sum(np.logical_and((x_pred==y),y))
    FN = np.sum(np.logical_and((x_pred!=y),np.logical_not(x_pred)))

    if TP != 0:
        return TP/(TP + FN)
    else:
        return 0

