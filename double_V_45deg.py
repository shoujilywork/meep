import sys
sys.path.append("/home/w/Desktop/meep-1.31.0/python")  # Add custom path
import meep as mp
import numpy as np
from pyevtk.hl import gridToVTK
import os
#os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# 引入 mpi4py 库
#from mpi4py import MPI 

# Simulation parameters
resolution = 30  # pixels/μm
x_um=12
y_um=12
z_um=8
cell_size = mp.Vector3(x_um, y_um, z_um)  # Simulation domain size (μm)
pml_layers = [mp.PML(1.0)]  # PML thickness

# Frequency range (GHz)
f_center = 3.0  # Center frequency
f_width = 1.0   # Pulse width

# Define the V-shaped dipole arms
arm_length = 0.8  # Length of each arm (μm)
arm_width = 0.2   # Width of the arms (μm)
angle = 45.0      # Angle between arms (degrees)

# Material properties (metal = perfect electric conductor)
metal = mp.metal  # PEC approximation
default_material=mp.Medium(epsilon=1.0)
# Create V-shaped dipole
def make_r_dipole(center):
    """Creates a V-shaped dipole centered at `center` and rotated by `rotation_angle`."""
    arm1 = mp.Block(
        center=center+ mp.Vector3(0.2,0,2),
        size=mp.Vector3(arm_length, arm_width, arm_width),
        material=metal,
        e1=mp.Vector3(np.sqrt(2)/2, 0, -np.sqrt(2)/2),
        e2=mp.Vector3(0, 1, 0),
        e3=mp.Vector3(np.sqrt(2)/2, 0, np.sqrt(2)/2)
    )
    return [arm1]
def make_l_dipole(center):
    """Creates a V-shaped dipole centered at `center` and rotated by `rotation_angle`."""
    arm1 = mp.Block(
        center=center+ mp.Vector3(-0.2,0,2),
        size=mp.Vector3(arm_length, arm_width, arm_width),
        material=metal,
        e1=mp.Vector3(np.sqrt(2)/2, 0, np.sqrt(2)/2),
        e2=mp.Vector3(0, 1, 0),
        e3=mp.Vector3(-np.sqrt(2)/2, 0, np.sqrt(2)/2)
    )
    return [arm1]
    
cone = mp.Cone(height=1.2,center=mp.Vector3(0, 0, 2),radius=1.4,radius2=0.4, material=metal, axis=mp.Vector3(0,0,1))

in_cone=mp.Cone(height=1.4,center=mp.Vector3(0, 0, 1.8),radius=1.3,radius2=0.2,material=default_material, axis=mp.Vector3(0,0,1))

hollow=[cone,in_cone]
# Create two crossed V-dipoles (90° apart)
geometry = hollow + make_r_dipole(mp.Vector3(0, 0, 0))+ make_l_dipole(mp.Vector3(0, 0, 0))

# Source (broadband dipole excitation)
sources = [
    mp.Source(
        mp.GaussianSource(frequency=f_center, fwidth=f_width),
        component=mp.Ez,
        center=mp.Vector3(0, 0, 2),
        size=mp.Vector3(0.05, .05, 0.05)  # Point source
    )
]

# Simulation setup
sim = mp.Simulation(
    cell_size=cell_size,
    geometry=geometry,
    default_material=default_material,
    sources=sources,
    resolution=resolution,
    boundary_layers=pml_layers,
    #split_chunks_evenly=True
)
#sim.use_output_directory("/home/w/meep_shared")
x=np.linspace(1,x_um*resolution,num=x_um*resolution)
y=np.linspace(1,y_um*resolution,num=y_um*resolution)
z=np.linspace(1,z_um*resolution,num=z_um*resolution)
def output_epsi(sim):
    filename="epsilon"
    #ex_data=sim.get_array(component=mp.Ex, center =mp.Vector3(),size=resolution*cell_size)
    #ey_data=sim.get_array(component=mp.Ey, center =mp.Vector3(),size=resolution*cell_size)
    epsi=sim.get_array(component=mp.Dielectric, center =mp.Vector3(),size=resolution*cell_size)
    gridToVTK(filename,x,y,z,pointData={"Epsilon":epsi})#,"Ey":ey_data,"Ex":ex_data})

def output_field(sim):
    t = sim.meep_time()
    format_t=str(t).zfill(5)
    filename="efield_"+format_t
    #ex_data=sim.get_array(component=mp.Ex, center =mp.Vector3(),size=resolution*cell_size)
    #ey_data=sim.get_array(component=mp.Ey, center =mp.Vector3(),size=resolution*cell_size)
    ez_data=sim.get_array(component=mp.Ez, center =mp.Vector3(),size=resolution*cell_size)
    gridToVTK(filename,x,y,z,pointData={"Ez":ez_data})#,"Ey":ey_data,"Ex":ex_data})
        
sim.run(mp.at_beginning(output_epsi),  
        mp.at_every(1, output_field),  
        until=3)

