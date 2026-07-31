import numpy as np


class AttrDict(dict): # selfmade parameter dictionary class
    def __init__(self, *args, **kwargs):
        super(AttrDict,self).__init__(*args,**kwargs)
        self.__dict__ = self

P = {}

def loadparams(scenario):
    return P[scenario]

##### load cortex fitting params
#---> Maris Sacha Coefficient
print("***Using Sacha Coefficients***")
RS_P=np.load('fitting_parameters/RS-cell0_CONFIG1_fit.npy')
FS_P=np.load('fitting_parameters/FS-cell_CONFIG1_fit.npy')

#---> Di Volo Repo
# print("***Using Di Volo Coefficient***")
# RS_P=[-0.04983106, 0.00506355, -0.02347012, 0.00229515, -0.00041053, 0.01054705, -0.03659253, 0.00743749, 0.00126506, -0.04072161]
# FS_P=[-0.05149122, 0.00400369, -0.00835201, 0.00024142, -0.00050706, 0.00143454, -0.01468669, 0.00450271, 0.00284722, -0.0153578]


#-------------------------------------------------------
# ===================Cortex ============================
#-------------------------------------------------------

params = {}

params['RS'] = AttrDict({
    'P' : RS_P,
    'order': ['Glutamate','GABA'],  # order in which the values in Q, T and E are arranged
    'Q' : [1.5e-9, 5e-9],
    'T' : [5e-3, 5e-3],
    'E' : [0, -80e-3],
    'Cm' : 200e-12,
    'El' : -64e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0,
    'b' : 0, 
    'Vm' : -60e-3,
    'Vr' : -65e-3,
    'Vth' : -50e-3,
    'Vcut' : -30e-3,
    'Delta_t' : 2e-3,
    'refractory_time' : 5e-3,
    'external_input' : 0.0
})

params['FS'] = AttrDict({
    'P' : FS_P,
    'order': ['Glutamate','GABA'],  # order in which the values in Q, T and E are arranged
    'Q' : [1.5e-9, 5e-9],
    'T' : [5e-3, 5e-3],
    'E' : [0, -80e-3],
    'Cm' : 200e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0,
    'b' : 0,
    'Vm' : -60e-3,
    'Vr' : -65e-3,
    'Vth' : -50e-3,
    'Vcut' : -30e-3,
    'Delta_t' : 0.5e-3,
    'refractory_time' : 5e-3,
    'external_input' : 0.0
})
P['RS-FS_awake'] = params


params = {}

params['RS'] = AttrDict({
    'P' : RS_P,
    'Nexc' : 8000,
    'Ninh' : 2000,
    'order': ['Glutamate','GABA'],  # order in which the values in Q, T and E are arranged
    'Q' : [1.5e-9, 5e-9],
    'T' : [5e-3, 5e-3],
    'E' : [0, -80e-3],
    'Cm' : 200e-12,
    'El' : -64e-3,
    'Gl' : 10e-9,
    'Tw' : 500e-3,
    'a' : 0,
    'b' : 200e-12,
    'Vm' : -60e-3,
    'Vr' : -65e-3,
    'Vth' : -50e-3,
    'Vcut' : -30e-3,
    'Delta_t' : 2e-3,
    'refractory_time' : 5e-3,
    'external_input' : 0.0
})

params['FS'] = AttrDict({
    'P' : FS_P,
    'Nexc' : 8000,
    'Ninh' : 2000,
    'order': ['Glutamate','GABA'],  # order in which the values in Q, T and E are arranged
    'Q' : [1.5e-9, 5e-9],
    'T' : [5e-3, 5e-3],
    'E' : [0, -80e-3],
    'Cm' : 200e-12,
    'El' : -65e-3,
    'Gl' : 10e-9,
    'Tw' : 300e-3,
    'a' : 0,
    'b' : 0,
    'Vm' : -60e-3,
    'Vr' : -65e-3,
    'Vth' : -50e-3,
    'Vcut' : -30e-3,
    'Delta_t' : 0.5e-3,
    'refractory_time' : 5e-3,
    'external_input' : 0.0
})
P['RS-FS_sleep'] = params
