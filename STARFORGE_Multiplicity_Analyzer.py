#!/usr/bin/env python
# coding: utf-8

#Importing relevant libraries
#from #platform #import #dist
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from get_sink_data import sinkdata
from pandas.core.common import flatten
import re
from scipy.special import gamma as Gamma,gammaincc as GammaInc_up, gammainc as GammaInc_low
import random 
import matplotlib
import matplotlib.patches as mpatches
import copy
import itertools
from scipy import stats, optimize
from scipy.spatial import cKDTree
#from sys import exit
from errno import EEXIST
from os import makedirs,path
from palettable.colorbrewer.qualitative import Set1_5, Set1_8, Dark2_8,Set3_12

#Convert the simulation time to Myrs
code_time_to_Myr = 978.461942384
#Convert AU to m (divide by this if you have a result in m)
m_to_AU = 149597870700.0
#Convert pc to AU
pc_to_AU = 206264.806
#Convert pc to m
pc_to_m = 3.08567758e16
#G in SI Units
G = 6.67e-11
#Solar Mass to kg
msun_to_kg = 1.9891e30
#seconds to day
s_to_day = 60 * 60*24
s_to_yr = 3.154e7

#The length for the given box plot
L = (4/3*np.pi)**(1/3)*10

global_minmass=0.08

#List of Plot Names for Plots() and Multi_Plot()
Plots_key = ['System Mass','Primary Mass','Mass Ratio','Semi Major Axis','Multiplicity','Multiplicity Time Evolution',
'Multiplicity Lifetime Evolution','Multiplicity vs Formation','YSO Multiplicity','Semi-Major Axis vs q']

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def adjust_font(lgnd=None, lgnd_handle_size=49, fig=None, ax_fontsize=14, labelfontsize=14,right = True,top = True, adjust_ticks=False):
    '''Change the font and handle sizes'''
    #Changing the legend size
    if not (lgnd is None):
        for handle in lgnd.legendHandles:
            handle.set_sizes([lgnd_handle_size])
    #Changing the axis and label text sizes
    if not (fig is None):
        ax_list = fig.axes
        for ax1 in ax_list:
            ax1.tick_params(axis='both', labelsize=ax_fontsize)
            ax1.set_xlabel(ax1.get_xlabel(),fontsize=labelfontsize)
            ax1.set_ylabel(ax1.get_ylabel(),fontsize=labelfontsize)
            ax1.minorticks_on()
            if adjust_ticks:
                ax1.tick_params(axis='both',which='both', direction='in',top=top,right=right)

# def adjust_font(lgnd=None, lgnd_handle_size=49, fig=None, ax_fontsize=14, labelfontsize=14,do_ticks=True ):
#     if not (lgnd is None):
#         for handle in lgnd.legendHandles:
#             handle.set_sizes([lgnd_handle_size])
#     if not (fig is None):
#         ax_list = fig.axes
#         for ax1 in ax_list:
#             ax1.tick_params(axis='both', labelsize=ax_fontsize)
#             ax1.set_xlabel(ax1.get_xlabel(),fontsize=labelfontsize)
#             ax1.set_ylabel(ax1.get_ylabel(),fontsize=labelfontsize)
#             if do_ticks:
#                 ax1.minorticks_on()
#                 ax1.tick_params(axis='both',which='both', direction='in',top=True,right=True)


def rolling_average(List,rolling_window = 10):
    '''Rolling Average function'''
    x = List
    N = rolling_window
    valid_ind1=(int)((N-1)/2);valid_ind2=len(x)-(int)((N-1)/2)
    #x_avg=np.convolve(x, np.ones((N,))/N, mode='same')
    x_avg=np.convolve(x, np.ones((N,))/N, mode='valid')
    x_return = np.array(x)
    x_return[valid_ind1:valid_ind2]=x_avg
    #return x_avg, valid_ind1, valid_ind2
    return x_return


def advanced_binning(data,target_size, min_bin_pop=0,max_bin_size=1e100,dx=1e-10,allow_empty_bins=True):
    #Give bin edges of target size while ensuring that we have at least min_bin_pop elements. No bins can be bigger than max_bin_size, even if the population is smaller than min_bin_pop but also at least 1  
    sortdata=np.sort(np.array(data)); N = len(sortdata)
    edges = [sortdata[0]-dx];  lastind = 0; #init
    for i,x in enumerate(sortdata):
        if i>0:
            bin_pop = i-lastind
            bin_val = (sortdata[i-1]+sortdata[i])/2
            bin_size = bin_val-edges[-1]
            if allow_empty_bins:
                while bin_size>2*max_bin_size:
                    edges.append(edges[-1]+max_bin_size); lastind=i
                    bin_size = bin_val-edges[-1] 
            if (i>N-min_bin_pop) and ( (sortdata[-1]-bin_val)<target_size ):
                edges.append(sortdata[-1])
                break
            elif ( (bin_pop>=min_bin_pop) and (bin_size>=target_size) ) or (bin_size>max_bin_size):
                edges.append(bin_val)
                lastind = i
    if edges[-1]<=sortdata[-1]:  edges[-1] = sortdata[-1]+dx
    return np.array(edges)


def snaps_to_time(n,file):
    '''Convert snapshots into time'''
    time_per_snap = (file[1].t-file[0].t)*code_time_to_Myr
    total_time = time_per_snap*n
    return total_time

def time_to_snaps(time,file):
    '''Convert a time into number of snapshots'''
    time_per_snap = (file[1].t-file[0].t)*code_time_to_Myr
    no_of_snaps = (time/time_per_snap).round(0)
    return no_of_snaps

def set_colors_and_styles(colors, styles, N, multiples=1, dark=False, cmap=None,sequential=False):
    if colors is None:
        if sequential and (N<=4):
            #colors_base = ['#a1dab4', '#41b6c4', '#2c7fb8','#253494']
            colors_base = [(161/255,218/255,180/255),(65/255,182/255,196/255),(44/255,127/255,184/255),(37/255,52/255,148/255) ]
            if N==1:
                colors = [colors_base[0]]
            elif N==2:
                colors = [colors_base[0],colors_base[-1]]
            else: 
                base_length = np.arange(len(colors_base))/(len(colors_base)-1)
                colors = [( np.interp(i/(N-1),base_length,np.array(colors_base)[:,0]) , np.interp(i/(N-1),base_length,np.array(colors_base)[:,1]), np.interp(i/(N-1),base_length,np.array(colors_base)[:,2])  ) for i in range(N)]
        else:
            if (N<=5):
                #colors=['k', 'g', 'r', 'b', 'm', 'brown', 'orange', 'cyan', 'gray', 'olive']
    #            if dark:
    #                colors=Dark1_5.mpl_colors[:N]
    #            else:
                    colors=Set1_5.mpl_colors[:N]
                    if N==3:
                        colors=['k', 'r', 'b']
            elif (N<=8):
                if dark:
                    colors=Dark2_8.mpl_colors[:N]
                else:
                    colors=Set1_8.mpl_colors[:N]
            elif (N<=12):
                    colors=Set3_12.mpl_colors[:N] 
            else:
                if cmap is None:
                    cm=matplotlib.cm.get_cmap()
                else:
                    cm=matplotlib.cm.get_cmap(cmap)
                colors=[cm(i/N) for i in range(N)]
    if styles is None:
        styles=np.full(N,'-')
    if multiples>1:
        colors_old=colors[:]; styles_old=styles[:]
        colors=[]; styles=[]
        for i in range(N):
            for j in range(multiples):
                colors.append(colors_old[i])
                styles.append(styles_old[j])
    return colors, styles

#Remove the brown dwarfs for a data
def Remove_Low_Mass_Stars(data,minmass = 0.08):
    '''
    Remove the Brown Dwarfs from the initial data file

    Inputs
    ----------
    data : list of sinkdata objects
    The file that needs to be filtered

    Parameters
    ----------
    minmass : float,int,optional
    The mass limit for Brown Dwarfs.

    Returns
    -------
    data : list of sinkdata objects
    The original file but the brown dwarfs are removed from the x,v,id,formation_time and m parameters

    Example
    -------
    M2e4_C_M_J_2e7 = Remove_Low_Mass_Stars(M2e4_C_M_J_2e7,minmass = 0.08)
    '''
    #Get the lowest snapshot in which there are any stars.
    lowest = 0
    for i in range(len(data)):
        if len(data[i].m)>0:
            break
        lowest = i
    #Change the data to remove the points corresponding to Brown Dwarfs
    for i in range(lowest,len(data)):
        data[i].x = data[i].x[data[i].m>minmass]
        data[i].v = data[i].v[data[i].m>minmass]
        data[i].id = data[i].id[data[i].m>minmass]
        data[i].formation_time = data[i].formation_time[data[i].m>minmass]
        for label in data[i].extra_data.keys():
            data[i].extra_data[label] = data[i].extra_data[label][data[i].m>minmass]
        data[i].m = data[i].m[data[i].m>minmass]

    return data

#Simple function that can return the closest index or value to another one
def closest(lst,element,param = 'value'):
    '''
    Get the closest value to a target element in a given list

    Inputs
    ----------
    lst : list,array
    The list to find the value from
        
    element: int,float
    The target element to find the closest value to in the list

    Parameters
    ----------
    param : string,optional
    The param can be set to value, which returns the closest value in the list or index, which returns the index of the closest value in the list. By default, it returns the value.

    Returns
    -------
    closest_ele : int,float
    Either the closest element (if param is value) or the index of the closest element(if param is index)

    Example
    -------
    1) closest([1,2,3,4],3.6,param = 'value')

    This returns the value of the element in the list closest to the given element. (4)

    2) closest([1,2,3,4],3.6,param = 'index')

    This returns the index of the list element closest to the given element. (3)
    '''
    lst = np.asarray(lst) 
    idx = (np.abs(lst - element)).argmin() 
    if param == 'value':
        return lst[idx]
    elif param == 'index':
        return idx

#Removing an item in a nested list
def nested_remove(L, x):
    'Remove an item from a list with multiple layers of nesting'
    if (x in L) :
        L.remove(x)
    else:
        for element in L:
            if type(element) is list:
                nested_remove(element, x)

def findsubsets(s, n):
    'Find all the subsets of a list (s) of length (n).'
    tuplelist = list(itertools.combinations(s, n))
    subset_list = []
    for i in range(len(tuplelist)):
        subset_list.append(list(tuplelist[i]))
    return subset_list

def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True

#This function finds the first snapshot that a star is in
def first_snap_finder(ide,file):
    'Find the first snapshot where an id is present in the file.'
    for i,j in enumerate(file):
            if ide in j.id:
                return i

#This finds the first snapshot that a star is in a certain mass range
def first_snap_mass_finder(ide,file,lower_limit,upper_limit):
    'Find the first snapshot where an id is present in the file in a certain mass range.'
    for i,j in enumerate(file):
        if ide in j.id and lower_limit<=j.m[j.id == ide][0]<=upper_limit:
            return i

def last_snap_finder(ID,file_data):
    'Find the last snapshot where an id is present'
    for i in range(-1,-len(file_data),-1):
        if ID in file_data[i].id:
            return i
    return 0 
 
def IGamma_up(k,n):
    '''The Upper Incomplete Gamma Function'''
    return GammaInc_up(k,n)*Gamma(k)

def IGamma_low(k,n):
    '''The lower Incomplete Gamma Function'''
    return GammaInc_low(k,n)*Gamma(k)

def sigmabinom(n,k):
    '''The Binomial Error Function'''
    if np.isnan(n) or np.isnan(k): return np.nan
    return np.sqrt((k*(n-k))/n**3)

def Psigma(n,k,limit=10):
    '''Complex Binomial Error Function'''
    if np.isnan(n) or np.isnan(k) or n==0: return np.nan
    # if n<limit:
    #     #return np.sqrt( (-Gamma(2+n)**2*Gamma(2+k)**2)/(Gamma(3+n)**2*Gamma(1+k)**2)+(Gamma(3+k)*Gamma(2+n))/(Gamma(1+k)*Gamma(4+n)) 
    # else:
    #     if k==n: k=n-1 #so that we get nonzero estimate
    #     if k==0: k=1 #so that we get nonzero estimate
    #     return sigmabinom(n,k) #use binomial apprximation
    return np.sqrt( (n-k+1)*(k+1)/((n+3)*(n+2)*(n+2)) )


def Lsigma(n,k,limit=50):
    '''Companion Frequency Error Function'''
    if np.isnan(n) or np.isnan(k) or n==0: return np.nan
    if k<limit:
        #return np.sqrt(-((Gamma(2+k)-IGamma_up(2+k,3*n))**2/(n**2*(Gamma(1+k)-IGamma_up(1+k,3*n))**2))+((Gamma(3+k)-IGamma_up(3+k,3*n))/(n**2*(Gamma(1+k)-IGamma_up(1+k,3*n)))))
        return np.sqrt( IGamma_low(k+3,3*n)/(n*n*IGamma_low(k+1,3*n)) - (IGamma_low(k+2,3*n)/(n*IGamma_low(k+1,3*n)))**2 )
    else:
        #Use Poisson to estimate
        #if k==0: k=1 #so that we get nonzero estimate
        return np.sqrt(k)/n




def load_files(filenames,brown_dwarfs = False):
    '''
    Initialize data from the provided filenames.

    Inputs
    ----------
    filenames : string (single file) or list of strings (multiple files)
    Input the file name(s) that you would like to initialize data from.

    Parameters
    ----------
    brown_dwarfs : bool,optional
    Removes all Brown Dwarfs from the file if False.

    Returns
    -------
    Files_List : array
    The files that were requested in an array.

    Example
    -------
    files = load_files([filename_1,filename_2,...],brown_dwarfs = False)
    '''
    if isinstance(filenames,str):
        filenames = [filenames] #If the input is a string
    Files_List = []
    for i in tqdm(range(len(filenames))):
        #Load the file for all files
        pickle_file = open(filenames[i]+str('.pickle'),'rb')
        data_file = pickle.load(pickle_file)
        pickle_file.close()
        if brown_dwarfs == False:
            filtered_data_file = Remove_Low_Mass_Stars(data_file)
        else:
            filtered_data_file = data_file
        Files_List.append(filtered_data_file) 

    return np.array(Files_List, dtype=object)

## This function calculates Binding Energy
def Binding_Energy(m1,m2,x1,x2,v1,v2,L = None):
    'Calculate the Binding Energy (in J) from given masses,positions and velocities. If using a box file, provide the lengths to modify it.'
    mu = (m1*m2)/(m1+m2)
    KE = msun_to_kg * 0.5 * ((v1[0]-v2[0])**2+(v1[1]-v2[1])**2 +(v1[2]-v2[2])**2) * mu
    dx = x1[0]-x2[0];dy = x1[1]-x2[1];dz = x1[2]-x2[2]
    #If the edge is periodic, we replace the long distance with the short distance in 1D.
    if not (L is None):
        if dx > L/2:
            dx = L - dx
        if dy > L/2:
            dy = L - dy
        if dz > L/2:
            dz = L - dz
    PE = (msun_to_kg**2/pc_to_m) * G*(m1+m2)*mu/np.sqrt(dx**2+dy**2+dz**2)
    E = KE - PE
    return E

#This function is able to calculate the binding energy matrix of a set of nodes
def Binding_Energy_Matrix(m,x,v,L = None):
    'Calculate the Binding Energy Matrix (in J) from a list of masses,positions and velocities. If using Box run, provide the edge lengths.'
    E_binding_mtx = np.zeros((len(m),len(m)))
    for i in range(len(m)):
        for j in range(i,len(m)):
                if i == j:
                    E = float('inf')
                else:
                    E = Binding_Energy(m[i],m[j],x[i],x[j],v[i],v[j],L = L) 
                E_binding_mtx[i][j] = E                
                E_binding_mtx[j][i] = E
    return E_binding_mtx


def find_bin_edges(xvals,target_binsize, minbinsize=0,verbose=False,dx=1e-10):
    #Find bin edges in array so that no points are close to the boundary
    x_sorted = np.sort(xvals)
    x_diff = np.diff(x_sorted)
    if not np.any(x_diff): 
        print("Warning: only identical coordinates present!"); minbinsize=0
    x_med = (x_sorted[1:]+x_sorted[0:-1])/2
    #Reduce minbinsize until we find a value that works
    while(1):
        valid_edges = (x_diff>=minbinsize)
        if len(valid_edges):
            break
        else:
            minbinsize /= 2.0
            if verbose: print("Using minbinsize=%g"%(minbinsize))
    #Brute-force but probably fine
    edges = [x_sorted[0]-dx]
    for edge_candidate in x_med[valid_edges]:
        if ( (edge_candidate-edges[-1])>=target_binsize ): #bin is at least target_binsize big
            edges.append(edge_candidate)    
    if len(edges)>1:
        edges[-1]=x_sorted[-1]+dx #expand last bin to ensure that we include all values
    else:
        edges.append(x_sorted[-1]+dx)
    return np.array(edges)

## This is done to speed up runtime and help remove outliers
# Since the pairs normally have a seperation of just 0.1 pc, we can just divide them into large chuncks of ~2-3 pc instead
def Splitting_Data(file,snapshot,seperation_param,L = None):
    '''
    Splitting the given files data into bins of a defined length. This would put the masses,positions, velocities and ids into different regions corresponding to
    the split regions.
    Inputs
    ----------
    file : list of sinkdata
    The data file to which the splitting will be applied
        
    snapshot: int
    The snapshot to which to apply splitting
        
    seperation_param : float,int
    The size of the box you want to split (in pc). If None, then no splitting is done
    Returns

    '''
    subregion_IDs = np.zeros_like(file[snapshot].id)
    if not (seperation_param is None): #we need to split
        # Defining a seperation parameter, the smaller the faster the program will run but the less accurate it will be
        #Doing the clustering using the digitize function for x,y and z seperately. Since the bins aren't zero indexed, they are 
        # all subtracted by one       
        boundary_ind = np.full(len(file[snapshot].m),False) #initialize as false
        if not(L is None): 
            boundary_ind = np.any( (file[snapshot].x>(L-seperation_param)) | (file[snapshot].x<seperation_param), axis=1) #check if any are in the boundary region
            subregion_IDs[boundary_ind] = -1 #assigning these to the boundary region
        xcoord=file[snapshot].x[~boundary_ind,0]; ycoord=file[snapshot].x[~boundary_ind,1]; zcoord=file[snapshot].x[~boundary_ind,2];    #taking non-boundary particles
        bins_x = find_bin_edges(xcoord,seperation_param, minbinsize=0.1*seperation_param)
        clusters_x_ind = np.digitize(xcoord,bins_x) - 1
        bins_y = find_bin_edges(ycoord,seperation_param, minbinsize=0.1*seperation_param)
        clusters_y_ind = np.digitize(ycoord,bins_y) - 1  
        bins_z = find_bin_edges(zcoord,seperation_param, minbinsize=0.1*seperation_param)
        clusters_z_ind = np.digitize(zcoord,bins_z) - 1 
        nbins = np.array([len(bins_x),len(bins_y),len(bins_z)])-1 #number of bins in each direction
        subregion_IDs[~boundary_ind] = clusters_x_ind + clusters_y_ind*nbins[0] + clusters_z_ind*nbins[0]*nbins[1]
      
    return subregion_IDs

## Most crucial function of the program (Program's runtime comes mainly from here)
def remove_and_replace(matrix,m,x,v,ids,L = None, qmin=0.0): 
    '''
    Find the minimum value in the Binding Energy Matrix and changes the masses, positions, velocities and ids of that node.

    Inputs
    ----------
    matrix : list, nested list
    The binding energy matrix to be analyzed.
        
    m: list, array
    The masses to be changed.
        
    x : list, array
    The positions to be changed.
        
    v : list, array
    The velocities to be changed.
        
    ids : list, array
    The ids to be changed.

    L: int,float,optional
    The size of periodic box, None if not periodic
    
    qmin: float, optional
    Minimum mass ratio allowed

    Returns
    -------

    new_matrix : list,nested list
        The new matrix with the rows and columns corresponding to the minimum element deleted.
        
    indexes : list
        The new ids with the previous min element's ids replaced with one list with both the ids.

    new_masses: list
        The new masses with the previous min element's masses replaced with one mass that is the sum of the ids.

    new_x : list
        The new ids with the previous min element's positions replaced with the position of the center of mass.

    new_v: list
        The new ids with the previous min element's velocities replaced with the velocity of the center of mass.

    Example
    -------
    1) new_matrix,indexes,new_masses,new_x,new_v = remove_and_replace(matrix,m,x,v,ids,L = 1)

    This is for a pre determined matrix,m,x,v and ids for a box file with side lengths 1.

    2) new_matrix,indexes,new_masses,new_x,new_v = remove_and_replace(matrix,m,x,v,ids)

    This is for a pre determined matrix,m,x,v and ids for a non-box file.

    '''
    #Optimized version of the remove_and_replace. This first finds the minimum element using a function(thus not looping through
    #the entire matrix) and then checks for two things, first, if its greater than 0 (unbound), then it stops the while loop or if the system it
    #is working on fits the criteria of 4 or less objects, it stops the while loop
    indexes = list(ids)
    most_bound_element = 0
    most_bound_element_indices = [0,0]
    comp_matrix = matrix
    flag = 0
    while flag == 0:
        most_bound_element = comp_matrix.min()
        if most_bound_element > 0:
            flag = 1
        most_bound_element_indices = list(np.unravel_index(comp_matrix.argmin(), comp_matrix.shape))
        most_bound_element_indices.sort()
        q = np.min(m[most_bound_element_indices[:2]])/np.max(m[most_bound_element_indices[:2]])
        if ( (len(list(flatten(list([indexes[most_bound_element_indices[0]],indexes[most_bound_element_indices[1]]])))))>4 ) or (q<qmin):
            comp_matrix[most_bound_element_indices[0]][most_bound_element_indices[1]] = np.inf
            comp_matrix[most_bound_element_indices[1]][most_bound_element_indices[0]] = np.inf
        else:
            flag = 1
    #If the most bound element is infinity, that means that we couldn't find any other
    #objects that fit the 4 objects criteria because we've replaced them all, so instead we return the input back
    if most_bound_element > 0:
        return matrix,ids,m,x,v
    else:
        small_i = most_bound_element_indices[0]
        big_i = most_bound_element_indices[1]
    #Creates the new indexes where the ids are joined
        indexes[small_i] = list([ids[small_i],ids[big_i]])
        del indexes[big_i]
    # Defining the new object's mass as the reduced mass of the initial ones, the coordinates as the CoM of the initial ones
    # and the velocity as the CoM velocity of the initial ones & also defining new mass,x and v from the CoM and total mass
    # and also getting the new x,v, and m from the total mass and CoM of x and v
        new_object_x = [0,0,0]
        new_object_v = [0,0,0]
        new_object_mass = (m[small_i]+m[big_i])
        
        new_x = list(x)
        new_v = list(v)
        new_masses = list(m)
        
        for i in range(0,3):
            new_object_x[i] = (((new_masses[big_i]*new_x[big_i][i])+(new_masses[small_i]*new_x[small_i][i]))/new_object_mass)
            new_object_v[i] = (((new_masses[big_i]*new_v[big_i][i])+(new_masses[small_i]*new_v[small_i][i]))/new_object_mass)
        
        new_x[small_i] = new_object_x
        new_x = np.delete(new_x,big_i,axis = 0)
        new_v[small_i] = new_object_v
        new_v = np.delete(new_v,big_i,axis = 0)
        new_masses[small_i] = new_object_mass
        new_masses = np.delete(new_masses,big_i)
    # Deleting one column and row corresponding to the smaller index, which makes the bigger index decrease by 1
        matrix = np.delete(matrix,most_bound_element_indices[1],0)
        matrix = np.delete(matrix,most_bound_element_indices[1],1)
        replace_indice = most_bound_element_indices[0]
    # Creating a new matrix and replacing the row and column with the replace index with the binding energies of the new object
    # and the other objects
        new_matrix = matrix
        Binding_Energy_row = []
        for i in range(len(new_masses)):
            if i == replace_indice:
                Binding_Energy_row.append(np.float('inf'))
            else: 
                Energy_of_these_objects = Binding_Energy(new_masses[i],new_object_mass,new_x[i],new_object_x,new_v[i],new_object_v,L = L)
                Binding_Energy_row.append(Energy_of_these_objects)
        new_matrix[replace_indice] = Binding_Energy_row
        for i in range(len(new_matrix)):
            new_matrix[i][replace_indice] = Binding_Energy_row[i]
    # Now the new matrix has the information we want. Thus, we'll want to send back the new masses, coordinates, velocities,
    #the new matrix and the indices of the removed objects
        return new_matrix,indexes,new_masses,new_x,new_v

# Since the previous function had the constraints already, we can make the while condition stop when there is no additional 
# clustering happening 
def constrained_remove_and_replace(binding_energy_matrix,ids,m,x,v,L = None):
    '''
    Perform the remove and replace until it is no longer possible (because there are no more bound systems of less than 4 stars).

    Inputs
    ----------
    binding_energy_matrix : list, nested list
    The binding energy matrix to be analyzed.

    ids : list, array
    The ids to be changed.
        
    m: list, array
    The masses to be changed.
        
    x : list, array
    The positions to be changed.
        
    v : list, array
    The velocities to be changed.

    L: int,float,optional
    The size of periodic box

    Returns
    -------
    ids: nested list
    The new ids containing the ids arranged by system.

    Example
    -------
    1) ids = constrained_remove_and_replace(matrix,ids,m,x,v,L = 1)

    This is for a pre determined matrix,m,x,v and ids for a box file with side lengths 1.

    2) ids = constrained_remove_and_replace(matrix,ids,m,x,v)

    This is for a pre determined matrix,m,x,v and ids for a non-box file.
    '''
    previous_ids = []
    # Wait till no more objects are clustered
    while len(previous_ids) != len(ids):
        previous_ids = list(ids)
        binding_energy_matrix, ids, m, x, v = remove_and_replace(binding_energy_matrix,m,x,v,ids,L = L)
    return ids

#This part of the program makes use of all the previous function definitions and leads to the result
def identify_bound_systems(file,snapshot_number,seperation_param = None,L = None):
    '''
    The main algorithm that can perform system assignment after splitting the data into boxes.
    Inputs
    ----------
    file : list of sinkdata.
    The input data that will be grouped into systems.

    snapshot_number : int
    The snapshot number to use.

    L: int,float,optional
    The size of periodic box

    Parameters
    ----------
    seperation_param : int, float
    The seperation of the boxes. By default, this is 2 pc and that is the separation of the pickle files. If set to None, no splitting is done

    Returns
    -------
    Result: list
    The new ids containing the ids arranged by system.

    Example
    -------
    1) Result = identify_bound_systems(M2e4_C_M_J_2e7,-1,seperation_param = 2,L = 1)

    This is for a box file with side lengths 1 and split every 2 pc.

    2) Result = identify_bound_systems(M2e4_C_M_J_2e7,-1,seperation_param = 2)

    This is for a non-box file that is split every 2 pc.
    '''
    #If the file has one or less stars, we just return the ids as is
    if len(file[snapshot_number].m) <=1:
        return file[snapshot_number].id
    #Otherwise, we perform out algorithm on the nodes.
    else:
        subregion_IDs = Splitting_Data(file,snapshot_number,seperation_param,L = L)
        Result = []; data = file[snapshot_number]
        for region_ID in tqdm(np.unique(subregion_IDs),position = 0,desc = 'Loop over subregions',leave= True ):
            ind = (subregion_IDs==region_ID)
            Binding = Binding_Energy_Matrix(data.m[ind],data.x[ind,:],data.v[ind,:],L = L)
            Result.append(constrained_remove_and_replace(Binding,data.id[ind],data.m[ind],data.x[ind,:],data.v[ind,:],L = L))
        return Result

#Calculating the semi major axis for the primary and secondary sub-systems of these systems
def smaxis(system):
    '''Calculate the semi major axis (in m) between the secondary and primary in a system.'''
    k = system #Don't want to rewrite all the ks
    if len(k.m) == 1: #Single star has no smaxis
        smaxis = 0
        return smaxis
    if len(k.m) == 2 and k.m[0] == k.m[1]:
        primary_id = k.ids[0]
        primary_mass = k.m[0]
        secondary_id = k.ids[1]
        secondary_mass = k.m[1]
        sec_ind = 1
    else:
        primary_id = k.primary_id
        primary_mass = k.primary
        secondary_id = k.ids[k.m == k.secondary]
        if isinstance(secondary_id,np.ndarray):
            secondary_id = secondary_id[0]
        secondary_mass = 0
    for i in range(len(k.m)):
        if k.m[i] < primary_mass and k.m[i]> secondary_mass:
            secondary_mass = k.m[i]
            sec_ind = i
    if k.no >= 2: #If there's two stars, only one possible semi major axis 
        vel_prim = k.v[np.argmax(k.m)]
        x_prim = k.x[np.argmax(k.m)]
        vel_sec = k.v[sec_ind]
        x_sec = k.x[sec_ind]
    if k.no == 3: #If there's three stars, only three configs: [[1,2],3] , [[1,3],2] and [[2,3],1]
        for i in k.structured_ids:
            if isinstance(i,list) and primary_id in i:
                if secondary_id in i:
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                else:
                    ind0 = (k.ids == i[0])
                    ind1 = (k.ids == i[1])
                    vel_prim = (k.m[ind0]*k.v[ind0] + k.m[ind1]*k.v[ind1])/(k.m[ind1]+k.m[ind0])
                    x_prim = (k.m[ind0]*np.array(k.x)[ind0] + k.m[ind1]*np.array(k.x)[ind1])/(k.m[ind1]+k.m[ind0])
                    primary_mass = (k.m[ind0]+k.m[ind1])[0]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
            elif isinstance(i,list) and secondary_id in i:
                if primary_id not in i:
                    ind0 = (k.ids == i[0])
                    ind1 = (k.ids == i[1])
                    vel_sec = (k.m[ind0]*k.v[ind0] + k.m[ind1]*k.v[ind1])/(k.m[ind1]+k.m[ind0])
                    x_sec = (k.m[ind0]*np.array(k.x)[ind0] + k.m[ind1]*np.array(k.x)[ind1])/(k.m[ind1]+k.m[ind0])
                    secondary_mass = k.m[ind0]+k.m[ind1]
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]

    if k.no == 4:# 4 is the most complex  [[1,2],[3,4]],[[1,3/4],[2,3/4]],[[[1,2],3/4],3/4],[[[1,3/4],2],3/4] or [[[1,3/4],3/4],2]
        struc_list = []
        for i in k.structured_ids:
            if isinstance(i,list):
                struc_list.append(len(i))
            else:
                struc_list.append(0)
        if sum(struc_list) == 4: #It is a binary of binaries
            for i in k.structured_ids:
                if primary_id in i and secondary_id in i:
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                else:
                    ind0 = (k.ids == i[0])
                    ind1 = (k.ids == i[1])
                    if primary_id in i and secondary_id not in i:
                        vel_prim = (k.m[ind0]*k.v[ind0] + k.m[ind1]*k.v[ind1])/(k.m[ind1]+k.m[ind0])
                        x_prim = (k.m[ind0]*np.array(k.x)[ind0] + k.m[ind1]*np.array(k.x)[ind1])/(k.m[ind1]+k.m[ind0])
                        primary_mass = (k.m[ind0]+k.m[ind1])
                    elif primary_id not in i and secondary_id in i:
                        vel_sec = (k.m[ind0]*k.v[ind0] + k.m[ind1]*k.v[ind1])/(k.m[ind1]+k.m[ind0])
                        x_sec = (k.m[ind0]*np.array(k.x)[ind0] + k.m[ind1]*np.array(k.x)[ind1])/(k.m[ind1]+k.m[ind0])
                        secondary_mass = k.m[ind0]+k.m[ind1]
        elif sum(struc_list) == 2: #It is hierarchial
            structure = []
            for i in k.structured_ids:
                substructure = []
                if isinstance(i,list):
                    for j in i:
                        if isinstance(j,list) and primary_id in j and secondary_id in j:
                            vel_prim = k.v[np.argmax(k.m)]
                            x_prim = k.x[np.argmax(k.m)]
                            vel_sec = k.v[sec_ind]
                            x_sec = k.x[sec_ind]
                        elif isinstance(j,list) and primary_id in j:
                            ind0 = (k.ids == j[0])
                            ind1 = (k.ids == j[1])
                            vel_prim = (k.m[ind0]*k.v[ind0] + k.m[ind1]*k.v[ind1])/(k.m[ind1]+k.m[ind0])
                            x_prim = (k.m[ind0]*np.array(k.x)[ind0] + k.m[ind1]*np.array(k.x)[ind1])/(k.m[ind1]+k.m[ind0])
                            primary_mass = k.m[ind0]+k.m[ind1]
                            substructure.append(42.0)
                        elif isinstance(j,list) and secondary_id in j:
                            ind0 = (k.ids == j[0])
                            ind1 = (k.ids == j[1])
                            vel_sec = (k.m[ind0]*k.v[ind0] + k.m[ind1]*k.v[ind1])/(k.m[ind1]+k.m[ind0])
                            x_sec = (k.m[ind0]*np.array(k.x)[ind0] + k.m[ind1]*np.array(k.x)[ind1])/(k.m[ind1]+k.m[ind0])
                            secondary_mass = k.m[ind0]+k.m[ind1]
                            substructure.append(24.0)
                        elif isinstance(j,list) and primary_id not in j and secondary_id not in j:
                            ind0 = (k.ids == j[0])
                            ind1 = (k.ids == j[1])
                            some_vel = (k.m[ind0]*k.v[ind0] + k.m[ind1]*k.v[ind1])/(k.m[ind1]+k.m[ind0])
                            some_x = (k.m[ind0]*np.array(k.x)[ind0] + k.m[ind1]*np.array(k.x)[ind1])/(k.m[ind1]+k.m[ind0])
                            some_mass = k.m[ind0]+k.m[ind1]
                            substructure.append(2.4)
                        elif isinstance(j,list) == False:
                            substructure.append(j)
                    structure.append(substructure)
                else:
                    structure.append(i)
            for stru in structure:
                if isinstance(stru,list) and 42.0 in stru and secondary_id in stru:
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                elif isinstance(stru,list) and 24.0 in stru and primary_id in stru:
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                elif isinstance(stru,list) and 42.0 in stru and secondary_id not in stru:
                    vel_prim = ((primary_mass*vel_prim)+((k.m[k.ids==np.array(stru)[np.array(stru)!=42.0]])*(k.v[k.ids==np.array(stru)[np.array(stru)!=42.0]])))/(primary_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=42.0]])
                    x_prim = (primary_mass*x_prim + k.m[k.ids==np.array(stru)[np.array(stru)!=42.0]]*np.array(k.x)[k.ids==np.array(stru)[np.array(stru)!=42.0]])/(primary_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=42.0]])
                    primary_mass = primary_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=42.0]]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                elif isinstance(stru,list) and 24.0 in stru and primary_id not in stru:
                    vel_sec = (secondary_mass*vel_sec + k.m[k.ids==np.array(stru)[np.array(stru)!=24.0]]*k.v[k.ids==np.array(stru)[np.array(stru)!=24.0]])/(secondary_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=24.0]])
                    x_sec = (secondary_mass*x_sec + k.m[k.ids==np.array(stru)[np.array(stru)!=24.0]]*np.array(k.x)[k.ids==np.array(stru)[np.array(stru)!=24.0]])/(secondary_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=24.0]])
                    secondary_mass = secondary_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=24.0]]
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                elif isinstance(stru,list) and 2.4 in stru and primary_id in stru:
                    vel_prim = ((some_mass*some_vel)+((k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]])*(k.v[k.ids==np.array(stru)[np.array(stru)!=2.4]])))/(some_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]])
                    x_prim = (some_mass*some_x + k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]]*np.array(k.x)[k.ids==np.array(stru)[np.array(stru)!=2.4]])/(some_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]])
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                    primary_mass = some_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]]
                elif isinstance(stru,list) and 2.4 in stru and secondary_id in stru:
                    vel_sec = ((some_mass*some_vel)+((k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]])*(k.v[k.ids==np.array(stru)[np.array(stru)!=2.4]])))/(some_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]])
                    x_sec = (some_mass*some_x + k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]]*np.array(k.x)[k.ids==np.array(stru)[np.array(stru)!=2.4]])/(some_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]])
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    secondary_mass = some_mass+k.m[k.ids==np.array(stru)[np.array(stru)!=2.4]]
                
    
    x_prim = list(flatten(x_prim))
    x_sec= list(flatten(x_sec))
    vel_prim = list(flatten(vel_prim))
    vel_sec= list(flatten(vel_sec))
    binding_energy = Binding_Energy(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
    mu = msun_to_kg*((primary_mass*secondary_mass)/(primary_mass+secondary_mass))
    eps = binding_energy/mu
    mu2 = msun_to_kg*G*(primary_mass+secondary_mass)
    smaxis = - (mu2/(2*eps))
    if isinstance(smaxis,np.ndarray):
        smaxis = smaxis[0]
    return smaxis

def semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec):
    '''Calculate the semimajor axis(in m) from given parameters'''
    x_prim = list(flatten(x_prim))
    x_sec= list(flatten(x_sec))
    vel_prim = list(flatten(vel_prim))
    vel_sec= list(flatten(vel_sec))
    binding_energy = Binding_Energy(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
    if binding_energy>0:
        return np.nan
    mu = msun_to_kg*((primary_mass*secondary_mass)/(primary_mass+secondary_mass))
    eps = binding_energy/mu
    mu2 = msun_to_kg*G*(primary_mass+secondary_mass)
    semiax = - (mu2/(2*eps))
    if isinstance(semiax,np.ndarray):
        semiax = semiax[0]
    return semiax

def eccentricity(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec):
    '''Calculate the eccentricity from given parameters'''
    x_prim = list(flatten(x_prim))
    x_sec= list(flatten(x_sec))
    vel_prim = list(flatten(vel_prim))
    vel_sec= list(flatten(vel_sec))
    binding_energy = Binding_Energy(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
    if binding_energy>0:
        return np.nan
    mu = msun_to_kg*((primary_mass*secondary_mass)/(primary_mass+secondary_mass))
    gravparam = msun_to_kg*G*(primary_mass+secondary_mass)
    dx = x_prim[0]-x_sec[0];dy = x_prim[1]-x_sec[1];dz = x_prim[2]-x_sec[2]
    dvx = vel_prim[0]-vel_sec[0];dvy = vel_prim[1]-vel_sec[1];dvz = vel_prim[2]-vel_sec[2]
    h = np.linalg.norm(np.cross([dx,dy,dz],[dvx,dvy,dvz])*pc_to_m)
    ecc = np.sqrt(1+2*binding_energy/mu/(gravparam**2)*(h**2))
    # a=semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec)
    # ecc_alt = np.sqrt( 1- (h**2)/a/gravparam )
    if isinstance(ecc,np.ndarray):
        ecc = ecc[0]

    return ecc

def ecc_smaxis_in_system(id1,id2,system):
    #semimajor_all, eccentricity_all, members_all = smaxis_all_func(system, return_all=True)
    tot_objects = 1e10; ecc = np.nan; semimajor=np.nan
    if (id2 is None) and (not (id1 is None)): id2=id1; id1=None
    for i, pairings in enumerate(system.orbits):
        both_in = ( (id1 in pairings[0]) and (id2 in pairings[1]) ) or ( (id1 in pairings[1]) and (id2 in pairings[0]) )
        one_in_other_none = (id1 is None) and ( (id2 in pairings[1])  or (id2 in pairings[0])  )                                                             
        if both_in or one_in_other_none:
           if (tot_objects>(len(pairings[0])+len(pairings[1]))):
               tot_objects = len(pairings[0])+len(pairings[1])
               ecc = system.ecc_all[i]
               semimajor = system.smaxis_all[i]
    return ecc, semimajor
                                                  


#Calculating the semi major axis for every possible configuration of these systems
def smaxis_all_func(system, return_all=False):
    '''Calculate the semimajor axis between all subsystems in a system'''
    k = system
    # if k.no!=len(k.m):
    #     print("Size mismatch", k.no, k.m, k.ids)
    if len(k.m) == 1: #Single star has no smaxis
        if return_all:
            return np.array([0]), np.array([0]), [k.ids]
        else:
            return 0

    primary_id = k.primary_id
    primary_mass = k.primary
    prim_ind = np.argmax(k.ids==k.primary_id)
    sec_ind = np.argmax((k.m <= primary_mass) & (k.ids != primary_id))
    secondary_id = k.ids[sec_ind]
    secondary_mass = k.m[sec_ind]
    
    semi_major_all = []
    eccentricity_all = []
    members_all = []
    
    if k.no == 2: #If there's two stars, only one possible semi major axis 
        vel_prim = k.v[prim_ind]
        x_prim = k.x[prim_ind]
        vel_sec = k.v[sec_ind]
        x_sec = k.x[sec_ind]
        semi_major_all.append(semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
        eccentricity_all.append(eccentricity(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
        members_all.append([[k.ids[0]],[k.ids[1]]])


    if k.no == 3: #If there's three stars, only three configs: [[1,2],3] , [[1,3],2] and [[2,3],1]
        tert_ind = np.argmax((k.m <= secondary_mass) & (k.ids != secondary_id))
        tert_mass = k.m[tert_ind]
        tert_x = k.x[tert_ind]
        tert_v = k.v[tert_ind]
        tert_id = k.ids[tert_ind]
        for i in k.structured_ids:
            if isinstance(i,list) and primary_id in i:
                if secondary_id in i:
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                    semi_major_all.append(semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
                    eccentricity_all.append(eccentricity(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
                    members_all.append([[primary_id],[secondary_id]])
                    
                    semi_major_all.append(semi_major_axis(primary_mass+secondary_mass,tert_mass,(x_prim*primary_mass+x_sec*secondary_mass)/(primary_mass+secondary_mass),tert_x,(vel_prim*primary_mass+vel_sec*secondary_mass)/(primary_mass+secondary_mass),tert_v))
                    eccentricity_all.append(eccentricity(primary_mass+secondary_mass,tert_mass,(x_prim*primary_mass+x_sec*secondary_mass)/(primary_mass+secondary_mass),tert_x,(vel_prim*primary_mass+vel_sec*secondary_mass)/(primary_mass+secondary_mass),tert_v))
                    members_all.append([[primary_id,secondary_id],[tert_id]])
                else:
                    ind0 = (k.ids == i[0])
                    ind1 = (k.ids == i[1])
                    semi_major_all.append(semi_major_axis(k.m[ind0][0],k.m[ind1][0],np.array(k.x)[ind0][0],np.array(k.x)[ind1][0],k.v[ind0][0],k.v[ind1][0]))
                    eccentricity_all.append(eccentricity(k.m[ind0][0],k.m[ind1][0],np.array(k.x)[ind0][0],np.array(k.x)[ind1][0],k.v[ind0][0],k.v[ind1][0]))
                    members_all.append([[k.ids[ind0][0]],[k.ids[ind1][0]]])
                    
                    vel_prim = (k.m[ind0][0]*k.v[ind0][0] + k.m[ind1][0]*k.v[ind1][0])/(k.m[ind1][0]+k.m[ind0][0])
                    x_prim = (k.m[ind0][0]*np.array(k.x)[ind0][0] + k.m[ind1][0]*np.array(k.x)[ind1][0])/(k.m[ind1][0]+k.m[ind0][0])
                    primary_mass = (k.m[ind0]+k.m[ind1])[0]
                    vel_sec = k.v[sec_ind]
                    x_sec = k.x[sec_ind]
                    semi_major_all.append(semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
                    eccentricity_all.append(eccentricity(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
                    members_all.append([[k.ids[ind0][0],k.ids[ind1][0]],[secondary_id]])
            elif isinstance(i,list) and secondary_id in i:
                if primary_id not in i:
                    ind0 = (k.ids == i[0])
                    ind1 = (k.ids == i[1])
                    semi_major_all.append(semi_major_axis(k.m[ind0][0],k.m[ind1][0],np.array(k.x)[ind0][0],np.array(k.x)[ind1][0],k.v[ind0][0],k.v[ind1][0]))
                    eccentricity_all.append(eccentricity(k.m[ind0][0],k.m[ind1][0],np.array(k.x)[ind0][0],np.array(k.x)[ind1][0],k.v[ind0][0],k.v[ind1][0]))
                    members_all.append([[k.ids[ind0][0]],[k.ids[ind1][0]]])
                    
                    vel_sec = (k.m[ind0]*k.v[ind0] + k.m[ind1]*k.v[ind1])/(k.m[ind1]+k.m[ind0])
                    x_sec = (k.m[ind0]*np.array(k.x)[ind0] + k.m[ind1]*np.array(k.x)[ind1])/(k.m[ind1]+k.m[ind0])
                    secondary_mass = k.m[ind0][0]+k.m[ind1][0]
                    vel_prim = k.v[np.argmax(k.m)]
                    x_prim = k.x[np.argmax(k.m)]
                    semi_major_all.append(semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
                    eccentricity_all.append(eccentricity(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
                    members_all.append([[k.ids[ind0][0],k.ids[ind1][0]],[primary_id]])

    if k.no == 4:# 4 is the most complex  [[1,2],[3,4]],[[1,3/4],[2,3/4]],[[[1,2],3/4],3/4],[[[1,3/4],2],3/4] or [[[1,3/4],3/4],2]
        struc_list = []
        for i in k.structured_ids:
            if isinstance(i,list):
                struc_list.append(len(i))
            else:
                struc_list.append(0)
        if sum(struc_list) == 4: #It is a binary of binaries
            ind0 = (k.ids == k.structured_ids[0][0])
            ind1 = (k.ids == k.structured_ids[0][1])
            semi_major_all.append(semi_major_axis(k.m[ind0][0],k.m[ind1][0],np.array(k.x)[ind0],np.array(k.x)[ind1],k.v[ind0],k.v[ind1]))
            eccentricity_all.append(eccentricity(k.m[ind0][0],k.m[ind1][0],np.array(k.x)[ind0],np.array(k.x)[ind1],k.v[ind0],k.v[ind1]))
            members_all.append([[k.ids[ind0][0]],[k.ids[ind1][0]]])
            
            vel_prim = (k.m[ind0]*k.v[ind0] + k.m[ind1]*k.v[ind1])/(k.m[ind1]+k.m[ind0])
            x_prim = (k.m[ind0]*np.array(k.x)[ind0] + k.m[ind1]*np.array(k.x)[ind1])/(k.m[ind1]+k.m[ind0])
            primary_mass = (k.m[ind0]+k.m[ind1])
            ind2 = (k.ids == k.structured_ids[1][0])
            ind3 = (k.ids == k.structured_ids[1][1])
            semi_major_all.append(semi_major_axis(k.m[ind2][0],k.m[ind3][0],np.array(k.x)[ind2],np.array(k.x)[ind3],k.v[ind2],k.v[ind3]))
            eccentricity_all.append(eccentricity(k.m[ind2][0],k.m[ind3][0],np.array(k.x)[ind2],np.array(k.x)[ind3],k.v[ind2],k.v[ind3]))
            members_all.append([[k.ids[ind2][0]],[k.ids[ind3][0]]])
            
            vel_sec = (k.m[ind2]*k.v[ind2] + k.m[ind3]*k.v[ind3])/(k.m[ind3]+k.m[ind2])
            x_sec = (k.m[ind2]*np.array(k.x)[ind2] + k.m[ind3]*np.array(k.x)[ind3])/(k.m[ind3]+k.m[ind2])
            secondary_mass = k.m[ind2]+k.m[ind3]
            semi_major_all.append(semi_major_axis(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
            eccentricity_all.append(eccentricity(primary_mass,secondary_mass,x_prim,x_sec,vel_prim,vel_sec))
            members_all.append([[k.ids[ind2][0],k.ids[ind1][0]],[k.ids[ind2][0],k.ids[ind3][0]]])
            
        elif sum(struc_list) == 2: #It is a hierarchial system
            for i in k.structured_ids:
                if isinstance(i,list):
                    for j in i:
                        if isinstance(j,list):
                            ind0 = k.ids == j[0]
                            ind1 = k.ids == j[1]
                            semi_major_all.append(semi_major_axis(k.m[ind0][0],k.m[ind1][0],k.x[ind0],k.x[ind1],k.v[ind0],k.v[ind1]))
                            eccentricity_all.append(eccentricity(k.m[ind0][0],k.m[ind1][0],k.x[ind0],k.x[ind1],k.v[ind0],k.v[ind1]))
                            members_all.append([[k.ids[ind0][0]],[k.ids[ind1][0]]])                            
                            primary_mass = k.m[ind0][0]+k.m[ind1][0]
                            vel_prim = (k.v[ind0]*k.m[ind0]+k.v[ind1]*k.m[ind1])/(k.m[ind0][0]+k.m[ind1][0])
                            x_prim = (k.x[ind0]*k.m[ind0]+k.x[ind1]*k.m[ind1])/(k.m[ind0][0]+k.m[ind1][0])
                        else:
                            outside_id = j
                    ind_out = k.ids == outside_id
                    semi_major_all.append(semi_major_axis(primary_mass,k.m[ind_out][0],x_prim,k.x[ind_out],vel_prim,k.v[ind_out]))
                    eccentricity_all.append(eccentricity(primary_mass,k.m[ind_out][0],x_prim,k.x[ind_out],vel_prim,k.v[ind_out]))
                    members_all.append([[k.ids[ind0][0],k.ids[ind1][0]],[k.ids[ind_out][0]]])  
                    x_prim = (primary_mass*x_prim+k.m[ind_out][0]*k.x[ind_out])/(primary_mass+k.m[ind_out][0])
                    vel_prim = (primary_mass*vel_prim+k.m[ind_out][0]*k.v[ind_out])/(primary_mass+k.m[ind_out][0])
                    primary_mass = primary_mass+k.m[ind_out][0] 
                else:
                    out_outside_id = i
                        
            ind_out2 = k.ids == out_outside_id
            semi_major_all.append(semi_major_axis(primary_mass,k.m[ind_out2][0],x_prim,k.x[ind_out2],vel_prim,k.v[ind_out2]))
            eccentricity_all.append(eccentricity(primary_mass,k.m[ind_out2][0],x_prim,k.x[ind_out2],vel_prim,k.v[ind_out2]))
            members_all.append([[k.ids[ind0][0],k.ids[ind1][0],k.ids[ind_out][0]],[k.ids[ind_out2][0]]])  
                       
        semi_major_all = np.array(semi_major_all)
        eccentricity_all = np.array(eccentricity_all)

    if return_all:
        return semi_major_all, eccentricity_all, members_all
    else:
        return semi_major_all

# Defining a class that contains the structured & non-structured ids, the masses, the snapshot number, the coordinates, the 
# velocities, the primary mass, the secondary mass, their ratio and the total mass of the system
class star_system:
    def  __init__(self,ids,n,data):
        self.structured_ids = ids #Saving the structured ids
        if isinstance(ids,list): #Flattening the ids
            self.ids = list(flatten(list(ids)))
            self.no = len(self.ids)
        else:
            self.ids = [ids]
            self.no = 1
        self.ids = np.array(self.ids,dtype=np.int64)
        index = np.array([np.argmax(data[n].id==sinkid) for sinkid in self.ids])
        self.m = data[n].m[index]
        self.x = data[n].x[index,:]
        self.v = data[n].v[index,:]
        self.snapshot_num = n #The snapshot number of the system
        self.tot_m = np.sum(self.m) #The total mass of the system
        primary_mass = max(self.m) #The primary (most massive) star in the system
        self.primary = primary_mass 
        self.primary_id = self.ids[np.argmax(self.m)] #The primary star's id
        secondary_mass = 0
        for i in range(len(self.m)):
            if self.m[i] < primary_mass and self.m[i]> secondary_mass:
                secondary_mass = self.m[i]
                sec_ind = i
        self.secondary = secondary_mass #The mass of the second most massive star
        self.mass_ratio = secondary_mass/primary_mass #The companion mass ratio (secondary/primary)
        #Note: The semi major axis is in m and is between the sub-systems with the primary & secondary
        self.smaxis = smaxis(self)
        self.smaxis_all, self.ecc_all, self.orbits = smaxis_all_func(self, return_all=True)
        #Get at formation density info
        self.init_star_vol_density = np.array([initial_local_density(ID,data,density = 'number',boxsize = None)[0] for ID in self.ids])
        self.init_star_mass_density = np.array([initial_local_density(ID,data,density = 'mass',boxsize = None)[0] for ID in self.ids])
        self.init_density = {'number': self.init_star_vol_density, 'mass': self.init_star_mass_density}
        if 'ProtoStellarAge' in data[n].extra_data.keys():
            #Get stellar evolution stage of stars
            self.stellar_evol_stages = np.array([data[n].extra_data['ProtoStellarStage'][data[n].id==ID] for ID in self.ids],dtype=np.int64)
            #Get formation times of stars
            self.formation_time_Myr = np.array([data[n].extra_data['ProtoStellarAge'][data[n].id==ID][0] for ID in self.ids])*code_time_to_Myr
        else:
            self.stellar_evol_stages = np.array([5 for ID in self.ids],dtype=np.int64)
            self.formation_time_Myr = np.array([data[first_snap_finder(ID,data)].t for ID in self.ids])*code_time_to_Myr
        self.age_Myr =  data[n].t*code_time_to_Myr - self.formation_time_Myr
        #Zero-age main-sequence (ZAMS) info
        self.ZAMS_age = ( data[n].t-np.array([data[n].formation_time[data[n].id==ID] for ID in self.ids]) ) * code_time_to_Myr
        for i,ID in enumerate(self.ids):
            if self.stellar_evol_stages[i]!=5:
                self.ZAMS_age[i] = -1
        #Get final masses of stars
        self.final_masses = []
        for ID in self.ids:
            last_snap = last_snap_finder(ID,data) #will almost always be -1, except for SNe
            self.final_masses.append(data[last_snap].m[data[last_snap].id==ID])
        self.final_masses = np.array(self.final_masses)
        #Mark multiplicity status of stars
        multiplicity_state = np.zeros_like(self.m)
        for i in range(len(self.m)):
            if self.m[i] == self.primary:
                multiplicity_state[i] = self.no-1
            else:
                multiplicity_state[i] = -1
        self.multip_state = multiplicity_state
        self.filter = np.nan
           
# Main Function of the program
def system_creation(file,snapshot_num,Master_File,seperation_param = None,read_in_result = False,L = None):
    '''
    The main function that does the system assignment(with/without splitting) and makes them star system objects.
    Inputs
    ----------
    file : list of sinkdata.
    The input data that will be grouped into systems.

    snapshot_number : int
    The snapshot number to use.
        
    Master_File : list of star system lists:
    The file containing already assigned systems to use if you don't want to apply algorithm again.
    
    seperation_param: float, distance in pc for subdivisions in which system assignment is carried out. If None, no subdivision is done

    L: int,float,optional
    The size of periodic box

    Parameters
    ----------
    read_in_result : bool
    Whether to read in the result from pickle files or do system assignment.

    Returns
    -------
    systems: list of star system objects
    The list of star system objects.

    Example
    -------
    1) systems = system_creation(M2e4_C_M_2e7,-1,M2e4_C_M_2e7_systems,read_in_result = True)

    This is the example of creating the systems for a non-boxed file with the systems already made (in the form of Master_File)

    2) systems = system_creation(M2e4_C_M_2e7,-1,M2e4_C_M_2e7_systems,read_in_result = True,L = L)

    This is the example of creating the systems for a boxed file with the systems already made (in the form of Master_File)

    3) systems = system_creation(M2e4_C_M_2e7,-1,'placeholder_text',read_in_result = False)

    This is the example of creating the systems for a non-boxed file where the systems aren't already made.

    '''
    systems = []
    #If you have the file, you can read it in from a premade Master_File, otherwise, you perform the algorithm
    if read_in_result == True:
        return Master_File[snapshot_num]
    else:
        Result = identify_bound_systems(file,snapshot_num,seperation_param = seperation_param,L = L)
   #Turn the id pairs into star system objects
    for i in tqdm(Result,desc = 'System Creation',position = 0,leave = True):
        if isinstance(i,list) or isinstance(i,np.ndarray):
            for j in i:
                systems.append(star_system(j,snapshot_num,file))
        else:
            systems.append(star_system(i,snapshot_num,file))
    return systems      

#This is an SFE finder, gives you the snapshot for an SFE by finding the closest value to the target SFE. It also gives you
#that SFE so that you know how close it is
def SFE_snapshot(file,SFE_param = 0.04):
    '''Figure out the closest snapshot in a file to a certain SFE value'''
    pickle_file = open(file,'rb')
    data = pickle.load(pickle_file) 
    pickle_file.close()
    initmass = np.float(re.search('M\de\d', file).group(0).replace('M',''))
    mass_sum = []
    for j in data:
        mass_sum.append(sum(j.m))
    SFE = []
    for j,l in enumerate(mass_sum):
        SFE.append(l/initmass)
    snap = closest(SFE,SFE_param,param='index')
    SFE_at = closest(SFE,SFE_param,param='value')
    print('SFE in the closest snapshot is '+str(SFE_at))
    return snap

#Finds the first snapshot with more than one star over a mass (i.e the first instance of creation of that mass)
def Mass_Creation_Finder(file,min_mass = 1):
    '''Find the first snapshot in a file which there is at least one star over a certain mass. '''
    snap = 0
    for i in range(len(file)):
        if len(file[i].m[file[i].m>min_mass]) != 0 and len(file[i].m>0):
            snap = i
            break
    return snap

def system_initialization(file,file_name,read_in_result = True,seperation_param = None,full_assignment = False,redo_all=False, snapshot_num = -1,L = None,starting_snap = 0, no_subdivision_for_last_snaps=1):
    '''
    This function initializes the systems for a given file.
    Inputs
    ----------
    file : list of sinkdata.
    The input data that will be grouped into systems.
        
    file_name: string
    The name of the file which will be matched to the system pickle file.

    Parameters
    ----------
    read_in_result : bool,optional
    Whether to read in the result from pickle files or do system assignment.
    full_assignment: bool,optional
    Whether to perform system assignment on all snapshots.
    
    redo_all: bool,optional
    If enabled all snapshots are recalculated even if previous system data exists
    
    seperation_param: float
    If not None the simulation volume is subdivided into boxes whose of seperation_param size and the systm assignment is carried out in them inependetly
        
    snapshot_num: int,optional
    The snapshot to perform assignment if you only want to do it for one snap (i.e full_assignment = False).

    L: int,float,optional
    The size of periodic box in pc.
    
    starting_snap: int,optional
    The first snap to perform system initialization
    
    no_subdivision_for_last_snaps: int, optional
    Sets the number of snapshots at the end of the simulation for which system assignment is done without subdivision, 0 by default

    Returns
    -------
    systems: list of star system objects,list of list of star system objects
    The list of all star systems from each snap or for one snap in the file. Note: The best variable name to save this is something like 'filename_systems'.

    Example
    -------
    1) M2e4_C_M_2e7_systems = system_initialization(M2e4_C_M_2e7,'M2e4_C_M_2e7',read_in_result = True)

    This is the example of creating the systems for a non-boxed file where the systems are already made.

    2) M2e4_C_M_2e7_systems = system_initialization(M2e4_C_M_2e7,'M2e4_C_M_2e7',read_in_result = True, L = L)

    This is the example of creating the systems for a boxed file of length L where the systems are already made.

    3) M2e4_C_M_2e7_systems = system_initialization(M2e4_C_M_2e7,'M2e4_C_M_2e7',read_in_result = False,snapshot_num = -1)

    This is the example of creating the systems for a single snapshot for a non-boxed file where the systems are not already made.

    4) M2e4_C_M_2e7_systems = system_initialization(M2e4_C_M_2e7,'M2e4_C_M_2e7',read_in_result = False,full_assignment = True)

    This is the example of creating all the systems for a non-boxed file where the systems are not already made.



    '''
    if read_in_result == True:
        infile = open(file_name+str('_Systems'),'rb')
        Master_File = pickle.load(infile)
        infile.close()
        return Master_File #Simply opening the pickle file
    elif read_in_result == False:
        if full_assignment == True:
            Result_List = []
            if redo_all or (not path.exists(file_name+str('_Systems'))):
                starting_snap=0
            if (not redo_all or starting_snap>0) and path.exists(file_name+str('_Systems')):
                print('Loading File '+file_name+str('_Systems'))
                infile = open(file_name+str('_Systems'),'rb')
                Master_File = pickle.load(infile)
                infile.close()
                print('File loaded')
                if starting_snap==0:
                    starting_snap = len(Master_File)
                for snap in range(0,starting_snap):
                    Result_List.append(Master_File[snap])
            for i in tqdm(range(starting_snap,len(file)),desc = 'Full Assignment',position = 0):
                print('Snapshot No: '+str(i)+'/'+str(len(file)-1))
                if i>=(len(file)-no_subdivision_for_last_snaps):
                    seperation_param_to_use = None
                else:
                    seperation_param_to_use = seperation_param
                Result_List.append(system_creation(file,i,Master_File = file,seperation_param=seperation_param_to_use,read_in_result = False,L = L))
            return Result_List #Returning the list of assigned systems
        else:#Returning just one snapshot
            return system_creation(file,snapshot_num,Master_File = file,seperation_param=seperation_param,read_in_result = False,L = L)

#This is a filter for minimum q for one snapshot
def q_filter_one_snap(systems,min_q = 0.1,filter_in_class = False):
    '''The q filter as applied to one snapshot'''
    Filtered_Master_File = copy.deepcopy(systems) #Creating a new copy of the master file
    for i,sys in enumerate(Filtered_Master_File):
        sys_orig = copy.deepcopy(sys)
        if sys.no>1:
            for ind, p_id in enumerate(sys_orig.ids):
                q = sys_orig.m[ind]/sys_orig.primary
                if q<min_q:
                    if sys.no == 4:
                        state = 0 #We need to see if its a [[1,2],[3,4]] or [1,[2,[3,4]]] system
                        for idd in sys.structured_ids:
                            if isinstance(idd,list):
                                state += len(idd)
                    remove_id = np.array(sys.ids)[sys.ids == p_id] #The id that we have to remove
                    sys.x = sys.x[sys.ids != p_id]
                    sys.v = sys.v[sys.ids != p_id]
                    sys.no -= 1
                    sys.age_Myr = sys.age_Myr[sys.ids != p_id]
                    sys.final_masses = sys.final_masses[sys.ids != p_id]
                    sys.formation_time_Myr = sys.formation_time_Myr[sys.ids != p_id]
                    sys.init_star_vol_density = sys.init_star_vol_density[sys.ids != p_id]
                    sys.init_star_mass_density = sys.init_star_mass_density[sys.ids != p_id]
                    sys.init_density = {'number': sys.init_star_vol_density, 'mass': sys.init_star_mass_density}
                    sys.stellar_evol_stages = sys.stellar_evol_stages[sys.ids != p_id]
                    sys.ZAMS_age = sys.ZAMS_age[sys.ids != p_id]
                    sys.multip_state = sys.multip_state[sys.ids != p_id] 
                    sys.m = sys.m[sys.ids != p_id]
                    sys.ids = sys.ids[sys.ids != p_id] #need to do this last
                    if sys.no == 1:
                        sys.secondary = 0
                        sys.smaxis = 0
                    else:
                        secondary_id = sys.ids[np.argmax(sys.m[sys.ids!=sys.primary_id])]
                        sys.secondary = sys.m[sys.ids==secondary_id][0]
                        sys.smaxis = ecc_smaxis_in_system(secondary_id, sys_orig.primary_id, sys_orig)[1]
                    sys.mass_ratio = sys.secondary/sys.primary
                    sys.tot_m = sum(sys.m)
                    if sys.no == 1:
                        sys.mass_ratio = 0
                        sys.secondary = 0 #Remove the secondary if the remaining star is solitary
                        sys.structured_ids = [sys.ids]
                    if sys.no == 2:
                        sys.structured_ids = list(sys.ids) #The secondary isn't going to be removed if there's 2 stars remaining
                    if sys.no == 3:
                        removed_list = copy.deepcopy(sys.structured_ids) 
                        checker = remove_id[0] #The remove ID is in an array so we make it single
                        checker = float(checker) #It is an np float so we make it a float
                        nested_remove(removed_list,checker)
                        if state == 4:
                            for index,value in enumerate(removed_list):
                                if isinstance(value,list):
                                    if len(value) == 1:
                                        removed_list[index] = value[0]
                        elif state == 2:
                            if len(removed_list) == 1:
                                removed_list = removed_list[0]
                            if len(removed_list) == 2:
                                for index,value in enumerate(removed_list):
                                    if isinstance(value,list) and len(value) == 1:
                                        removed_list[index] = value[0]
                                    elif isinstance(value,list) and len(value) == 2:
                                        removed_list[index] = list(flatten(value)) 
                        sys.structured_ids = removed_list
                    kept_smaxes = []; kept_ecc = []; kept_orbits=[] 
                    for i in range(len(sys.orbits)):
                        if not ( ( (p_id in sys.orbits[i][0]) and len(sys.orbits[i][0])==1 ) or ( (p_id in sys.orbits[i][1]) and len(sys.orbits[i][1])==1 ) ):
                            kept_smaxes.append(sys.smaxis_all[i]); kept_ecc.append(sys.ecc_all[i]); 
                            sys_orbits = copy.deepcopy(sys.orbits[i])
                            if (p_id in sys.orbits[i][0]): sys_orbits[0].remove(p_id)
                            elif (p_id in sys.orbits[i][1]): sys_orbits[1].remove(p_id)
                            kept_orbits.append(sys_orbits)
                    sys.smaxis_all = np.array(kept_smaxes)
                    sys.ecc_all = np.array(kept_ecc)
                    sys.orbits = kept_orbits
                    Filtered_Master_File[i] = sys
    return Filtered_Master_File

#This function applies the q filter to all snapshots
def q_filter(Master_File):
    '''Applying the q filter to an entire file'''
    New_Master_File = []
    for i in tqdm(Master_File,position = 0,desc = 'Full File q filter loop'):
        appending = q_filter_one_snap(i)
        New_Master_File.append(appending)
    return New_Master_File

#Modified version of the q filter to account for solar type star incompleteness in Raghavan
def Raghavan_filter_one_snap(systems):
    qlim1 = [0, 0.1]; completeness1 = 0.0; semimajor_limits1=[0, 1e10]
    qlim2 = [0.1, 0.5]; completeness2 = 0.0; semimajor_limits2=[0, 30]
    qlim3 = [0.1, 0.2]; completeness3 = 0.0; semimajor_limits3=[150, 400]
    '''The q filter as applied to one snapshot'''
    Filtered_Master_File = copy.deepcopy(systems) #Creating a new copy of the master file
    for i,sys in enumerate(Filtered_Master_File):
        sys_orig = copy.deepcopy(sys)
        if sys.no>1:
            for ind, p_id in enumerate(sys_orig.ids):
                q = sys_orig.m[ind]/sys_orig.primary
                ###a = sys.smaxis/m_to_AU #not at all accurate for multiple systems but Solar type stars rarely have higher order systems
                a = ecc_smaxis_in_system(p_id, sys_orig.primary_id, sys_orig)[1]/m_to_AU #this needs to be relative to primary
                remove_flag = False
                if (q<=qlim1[1]) and (q>=qlim1[0]) and (a<=semimajor_limits1[1]) and (a>=semimajor_limits1[0]):
                    remove_flag = np.random.rand()<(1-completeness1)
                elif  (q<=qlim2[1]) and (q>=qlim2[0]) and (a<=semimajor_limits2[1]) and (a>=semimajor_limits2[0]):
                    remove_flag = np.random.rand()<(1-completeness2)
                elif  (q<=qlim3[1]) and (q>=qlim3[0]) and (a<=semimajor_limits3[1]) and (a>=semimajor_limits3[0]):
                    remove_flag = np.random.rand()<(1-completeness3)
                if remove_flag:
                    if sys.no == 4:
                        state = 0 #We need to see if its a [[1,2],[3,4]] or [1,[2,[3,4]]] system
                        for idd in sys.structured_ids:
                            if isinstance(idd,list):
                                state += len(idd)
                    remove_id = np.array(sys.ids)[sys.ids == p_id] #The id that we have to remove
                    sys.x = sys.x[sys.ids != p_id]
                    sys.v = sys.v[sys.ids != p_id]
                    sys.no -= 1
                    sys.age_Myr = sys.age_Myr[sys.ids != p_id]
                    sys.final_masses = sys.final_masses[sys.ids != p_id]
                    sys.formation_time_Myr = sys.formation_time_Myr[sys.ids != p_id]
                    sys.init_star_vol_density = sys.init_star_vol_density[sys.ids != p_id]
                    sys.init_star_mass_density = sys.init_star_mass_density[sys.ids != p_id]
                    sys.init_density = {'number': sys.init_star_vol_density, 'mass': sys.init_star_mass_density}
                    sys.stellar_evol_stages = sys.stellar_evol_stages[sys.ids != p_id]
                    sys.ZAMS_age = sys.ZAMS_age[sys.ids != p_id]
                    sys.multip_state = sys.multip_state[sys.ids != p_id] 
                    sys.m = sys.m[sys.ids != p_id]
                    sys.ids = sys.ids[sys.ids != p_id] #need to do this last
                    if sys.no == 1:
                        sys.secondary = 0
                        sys.smaxis = 0
                    else:
                        secondary_id = sys.ids[np.argmax(sys.m[sys.ids!=sys.primary_id])]
                        sys.secondary = sys.m[sys.ids==secondary_id][0]
                        sys.smaxis = ecc_smaxis_in_system(secondary_id, sys_orig.primary_id, sys_orig)[1]
                    sys.mass_ratio = sys.secondary/sys.primary
                    sys.tot_m = sum(sys.m)
                    if sys.no == 1:
                        sys.mass_ratio = 0
                        sys.secondary = 0 #Remove the secondary if the remaining star is solitary
                        sys.structured_ids = [sys.ids]
                    if sys.no == 2:
                        sys.structured_ids = list(sys.ids) #The secondary isn't going to be removed if there's 2 stars remaining
                    if sys.no == 3:
                        removed_list = copy.deepcopy(sys.structured_ids) 
                        checker = remove_id[0] #The remove ID is in an array so we make it single
                        checker = float(checker) #It is an np float so we make it a float
                        nested_remove(removed_list,checker)
                        if state == 4:
                            for index,value in enumerate(removed_list):
                                if isinstance(value,list):
                                    if len(value) == 1:
                                        removed_list[index] = value[0]
                        elif state == 2:
                            if len(removed_list) == 1:
                                removed_list = removed_list[0]
                            if len(removed_list) == 2:
                                for index,value in enumerate(removed_list):
                                    if isinstance(value,list) and len(value) == 1:
                                        removed_list[index] = value[0]
                                    elif isinstance(value,list) and len(value) == 2:
                                        removed_list[index] = list(flatten(value)) 
                        sys.structured_ids = removed_list
                    kept_smaxes = []; kept_ecc = []; kept_orbits=[] 
                    for i in range(len(sys.orbits)):
                        if not ( ( (p_id in sys.orbits[i][0]) and len(sys.orbits[i][0])==1 ) or ( (p_id in sys.orbits[i][1]) and len(sys.orbits[i][1])==1 ) ):
                            kept_smaxes.append(sys.smaxis_all[i]); kept_ecc.append(sys.ecc_all[i]); 
                            sys_orbits = copy.deepcopy(sys.orbits[i])
                            if (p_id in sys.orbits[i][0]): sys_orbits[0].remove(p_id)
                            elif (p_id in sys.orbits[i][1]): sys_orbits[1].remove(p_id)
                            kept_orbits.append(sys_orbits)
                    sys.smaxis_all = np.array(kept_smaxes)
                    sys.ecc_all = np.array(kept_ecc)
                    sys.orbits = kept_orbits
                    Filtered_Master_File[i] = sys
    return Filtered_Master_File


def simple_filter_one_system(system,Master_File,comparison_snapshot = -2):
    'Working the simple filter onto one system'
    was_primary_there = False
    new_system = copy.deepcopy(system)
    for previous_sys in Master_File[comparison_snapshot]:
        if system.primary_id == previous_sys.primary_id:
            previous_target_system = previous_sys
            was_primary_there = True
    if system.no == 1:
        return new_system
    if was_primary_there == False: #If primary wasn't there, we make a new system of just the primary
        new_system.no = 1
        new_system.ids = np.array([new_system.primary_id])
        new_system.secondary = 0
        new_system.x = new_system.x[new_system.m == new_system.primary]
        new_system.v = new_system.v[new_system.m == new_system.primary]
        new_system.age_Myr = new_system.age_Myr[new_system.m == new_system.primary]
        new_system.final_masses = new_system.final_masses[new_system.m == new_system.primary]
        new_system.formation_time_Myr = new_system.formation_time_Myr[new_system.m == new_system.primary]
        new_system.init_star_vol_density= new_system.init_star_vol_density[new_system.m == new_system.primary]
        new_system.init_star_mass_density = new_system.init_star_mass_density[new_system.m == new_system.primary]
        new_system.init_density = {'number': new_system.init_star_vol_density, 'mass': new_system.init_star_mass_density}
        new_system.stellar_evol_stages = new_system.stellar_evol_stages[new_system.m == new_system.primary]
        new_system.ZAMS_age = new_system.ZAMS_age[new_system.m == new_system.primary]
        new_system.multip_state = new_system.multip_state[new_system.m == new_system.primary]
        new_system.m = np.array([new_system.primary])
        new_system.mass_ratio = 0
        new_system.tot_m = new_system.primary
        new_system.structured_ids = np.array([new_system.primary_id])
        new_system.smaxis = 0
        smaxis_all, ecc_all, orbits = smaxis_all_func(new_system, return_all=True)
        new_system.smaxis_all = smaxis_all; new_system.ecc_all = ecc_all; new_system.orbits = orbits
        return new_system
    for ides in system.ids: #Checking all the ids in the snap
        if ides not in previous_target_system.ids and ides != system.primary_id: #If any of the companions arent there
            if system.no == 4:
                state = 0 #We need to see if its a [[1,2],[3,4]] or [1,[2,[3,4]]] system
                for idd in system.structured_ids:
                    if isinstance(idd,list):
                        state += len(idd)
            keep_ind = (np.array(new_system.ids) != ides)
            new_system.ids = new_system.ids[keep_ind]
            new_system.x = new_system.x[keep_ind]
            new_system.v = new_system.v[keep_ind]
            new_system.no -= 1
            new_system.age_Myr = new_system.age_Myr[keep_ind]
            new_system.final_masses = new_system.final_masses[keep_ind]
            new_system.formation_time_Myr = new_system.formation_time_Myr[keep_ind]
            new_system.init_star_vol_density= new_system.init_star_vol_density[keep_ind]
            new_system.init_star_mass_density = new_system.init_star_mass_density[keep_ind]
            new_system.init_density = {'number': new_system.init_star_vol_density, 'mass': new_system.init_star_mass_density}
            new_system.stellar_evol_stages = new_system.stellar_evol_stages[keep_ind]
            new_system.ZAMS_age = new_system.ZAMS_age[keep_ind]
            new_system.multip_state = new_system.multip_state[keep_ind]
            new_system.m = new_system.m[keep_ind]
            new_system.tot_m = sum(new_system.m)
            if new_system.no == 1:
                new_system.mass_ratio = 0
                new_system.secondary = 0 #Remove the secondary if the remaining star is solitary
                new_system.structured_ids = np.array([system.primary_id])
            if new_system.no == 2:
                new_system.structured_ids = np.array(list(new_system.ids))
                secondary = 0
                for j in new_system.m:
                    if j < new_system.primary and j > secondary:
                        secondary = j
                new_system.secondary = secondary
                new_system.mass_ratio = secondary/new_system.primary
            if new_system.no == 3:
                removed_list = copy.deepcopy(new_system.structured_ids) 
                nested_remove(removed_list,float(ides))
                if state == 4:
                    for index,value in enumerate(removed_list):
                        if isinstance(value,list):
                            if len(value) == 1:
                                removed_list[index] = value[0]
                elif state == 2:
                    if len(removed_list) == 1:
                        removed_list = removed_list[0]
                    if len(removed_list) == 2:
                        for index,value in enumerate(removed_list):
                            if isinstance(value,list) and len(value) == 1:
                                removed_list[index] = value[0]
                            elif isinstance(value,list) and len(value) == 2:
                                removed_list[index] = list(flatten(value))
                new_system.structured_ids = removed_list
                secondary = 0
                for j in new_system.m:
                    if j < new_system.primary and j > secondary:
                        secondary = j
                new_system.secondary = secondary
                new_system.mass_ratio = secondary/system.primary
                
                
                kept_smaxes = []; kept_ecc = []; kept_orbits=[] 
                for i in range(len(new_system.orbits)):
                    if not ( ( (ides in new_system.orbits[i][0]) and len(new_system.orbits[i][0])==1 ) or ( (ides in new_system.orbits[i][1]) and len(new_system.orbits[i][1])==1 ) ):
                            kept_smaxes.append(new_system.smaxis_all[i]); kept_ecc.append(new_system.ecc_all[i]); 
                            sys_orbits = copy.deepcopy(new_system.orbits[i])
                            if (ides in new_system.orbits[i][0]): sys_orbits[0].remove(ides)
                            elif (ides in new_system.orbits[i][1]): sys_orbits[1].remove(ides)
                            kept_orbits.append(sys_orbits)
                new_system.smaxis_all = np.array(kept_smaxes)
                new_system.ecc_all = np.array(kept_ecc)
                new_system.orbits = kept_orbits
                new_system.smaxis = smaxis(new_system)
    return new_system

def full_simple_filter(Master_File,file,selected_snap = -1,long_ago = 0.1,no_of_orbits = 2,filter_in_class = True):
    if filter_in_class is True:
        filtered_systems = []
        for system_no,system in enumerate(Master_File[selected_snap]):
            filtered_systems.append(system.filter['time'])
        return filtered_systems
    else:
        if file[selected_snap].t*code_time_to_Myr<long_ago:
            #We cant look at a snapshot before 0.5 Myr 
            #print('The selected snapshot is too early to use')
            return Master_File[selected_snap]
        snap_1 = selected_snap-1
        snap_2 = selected_snap-2
        times = []
        filtered_systems = copy.deepcopy(Master_File[selected_snap])
        for i in file:
            times.append(i.t*code_time_to_Myr)
        long_ago_snap = closest(times,file[selected_snap].t*code_time_to_Myr - long_ago,param = 'index')
        for system_no,system in enumerate(Master_File[selected_snap]):
            result_1 = simple_filter_one_system(system,Master_File,comparison_snapshot=snap_1)
            result_2 = simple_filter_one_system(result_1,Master_File,comparison_snapshot=snap_2)
            orbital_period_check = (no_of_orbits*2*np.pi*np.sqrt(((smaxis(system))**3)/(6.67e-11*system.primary*msun_to_kg)))/(s_to_yr*1e6)
            orbital_period_snap = closest(times,file[selected_snap].t*code_time_to_Myr - orbital_period_check,param = 'index')
            if orbital_period_check > long_ago:
                snap_3 = orbital_period_snap
            else:
                snap_3 = long_ago_snap
            result_3 = simple_filter_one_system(result_2,Master_File,comparison_snapshot=snap_3)
            filtered_systems[system_no] = result_3
        return filtered_systems

def get_q_and_time(systems, Master_File):
    systems_q =  q_filter_one_snap(systems)
    Master_File_q = q_filter(Master_File); Master_File=0
    filtered_systems = []
    for system_no,system in enumerate(systems_q):
        filtered_systems.append(simple_filter_one_system(system,Master_File_q))
    return filtered_systems

def default_GMC_R(initmass = 2e4):
    '''Make the default R 10 pc for a GMC of 2e4 '''
    return 10

def file_properties(filename,param = 'm'):
    '''Get the initial properties of the cloud from the file name.'''
    #Lets get the initial gas mass for each, which we can only get from the name
    f = filename
    initmass=np.float(re.search('M\de\d', f).group(0).replace('M',''))
    if re.search('R\d', f) is None:
        R=default_GMC_R(initmass)
    else:
        R=np.float(re.search('R\d\d*', f).group(0).replace('R',''))
    if re.search('alpha\d', f) is None:
        alpha=2.0
    else:
        alpha=np.float(re.search('alpha\d*', f).group(0).replace('alpha',''))
    if 'Res' in f:
        npar=np.float(re.search('Res\d*', f).group(0).replace('Res',''))**3
    else:
        npar=np.float(re.search('_\de\d', f).group(0).replace('_',''))
    if param == 'm':
        return initmass
    elif param == 'r':
        return R
    elif param == 'alpha':
        return alpha
    elif param == 'res':
        return npar

def t_ff(mass,R):
    '''Calculate the freefall time'''
    G_code=4325.69
    tff = np.sqrt(3.0*np.pi/( 32*G_code*( mass/(4.0*np.pi/3.0*(R**3)) ) ) )
    return tff

def time_array(file,unit = 'code',t_ff = 1):
    time = []
    for i in file:
        time.append(i.t)
    time = np.array(time)
    if unit == 'Myr':
        time = time*code_time_to_Myr
    elif unit == 't_ff':
        time = time*t_ff
    return time

def initial_local_density(ID,file,des_ngb = 32,density = 'number',boxsize = None):
    '''Find the initial number of stars around a selected star, within a distance, when it was first formed'''
    first_snap = first_snap_finder(ID,file)
    #formation_pos = file[first_snap].x[file[first_snap].id == ID]
    if 1<len(file[first_snap].m)<=des_ngb:
        des_ngb = len(file[first_snap].m)
    elif len(file[first_snap].m) == 1:
        return np.nan,file[first_snap].t*code_time_to_Myr   
    x = file[first_snap].x#position
    m = file[first_snap].m #mass
    ids = file[first_snap].id #ids
    #Hack to check for Box size, should be replaced by propagating down the boxsize parameter
    boxsize = np.max(x)+1e-5
    #print("Estimated box size: %g pc. This is a temporary hack, in the future this should be a parameter."%(boxsize))
    tree = cKDTree(x, boxsize=boxsize)
    ngbdist, ngb = tree.query(x, des_ngb) #note that it will count the particle itself as its a neighbor
    #ngb_ids =  ids[ngb]
    ngb_vol =  4*np.pi/3 * (ngbdist[:,-1]**3)
    ngb_vol_density = des_ngb / ngb_vol
    ngb_mass_density = np.sum(m[ngb],axis=1) / ngb_vol
    if density == 'number':
        dens = ngb_vol_density[file[first_snap].id == ID]
    elif density == 'mass':
        dens = ngb_mass_density[file[first_snap].id == ID]
    
    return dens[0],file[first_snap].t*code_time_to_Myr

def new_stars_count(file,plot = True,time = True,all_stars = False,lower_limit = 0,upper_limit = 10000,rolling_avg = False,rolling_window_Myr = 0.1):
    '''
    The count of new stars or all stars of a certain mass range formed at different snapshots.
    Inputs
    ----------
    file : list of sinkdata.
    The input data that will be grouped into systems.

    Parameters
    ----------
    plot : bool,optional
    Whether to plot the number of stars.

    time: bool,optional
    Whether to have snapshot number or time as the x axis.
        
    all_stars: bool,optional
    Whether to calculate all stars at a snapshot or just the new stars.

    lower_limit: int,float,optional
    The lower limit of the mass range

    upper_limit: int,float,optional
    The upper limit of the mass range

    rolling_avg: bool,optional
    Whether to use a rolling average.

    rolling_window_Myr: int,float,optional
    The time to use in a rolling average. [in Myr]

    Returns
    -------
    no_of_stars: list
    Either the list of number of new stars or the number of total stars at each snapshot.

    Example
    -------
    1) new_stars_count(M2e4_C_M_2e7,time = True)
    Plotting the new stars count over time.

    2) new_stars_count(M2e4_C_M_2e7,all_stars = True)
    Plotting the total stars over time.

    3) new_stars_count(M2e4_C_M_2e7,time = True,lower_limit = 0,upper_limit = 1)
    Plotting the new stars count between 0 and 1 solar mass over time.
    '''
    no_new_stars = []
    times = []
    previous = 0
    new = 0
    no_of_stars = []
    rolling_window = int((int(time_to_snaps(rolling_window_Myr,file))//2)*2+1)
    for i in file:
        new = len(i.m[(lower_limit<=i.m) & (upper_limit>=i.m)])
        no_new_stars.append(new-previous)
        previous = new
        times.append(i.t*code_time_to_Myr)
        no_of_stars.append(new)
    if all_stars == False:
        if plot == True and time == True:
            if rolling_avg is False:
                plt.plot(times,no_new_stars)
            else:
                plt.plot(rolling_average(times,rolling_window = rolling_window),rolling_average(no_new_stars,rolling_window = rolling_window))
            plt.xlabel('Time[Myr]')
            plt.ylabel('Number of New Stars')
        elif plot == True and time == False:
            plt.plot(range(len(no_new_stars)),no_new_stars)
            plt.xlabel('Snapshot No')
            plt.ylabel('Number of New Stars')
        elif plot == False:
            return no_new_stars
    elif all_stars == True:
        if plot == True and time == True:
            if rolling_avg is True:
                plt.plot(times,rolling_average(no_of_stars,rolling_window = rolling_window))
            else:
                plt.plot(times,no_of_stars)
            plt.xlabel('Time[Myr]')
            plt.ylabel('Number of Stars')
        elif plot == True and time == False:
            plt.plot(range(len(no_new_stars)),no_of_stars)
            plt.xlabel('Snapshot No')
            plt.ylabel('Number of Stars')
        elif plot == False:
            return no_of_stars

def formation_time_histogram(file,systems = None,upper_limit=1.3,lower_limit = 0.7,target_mass = None,filename = None,plot = True,min_time_bin = 0.2,only_primaries_and_singles = False,full_form_times = False, label = None):
    '''
    Create or return a histogram of the formation times of stars in the given mass range.
    
    Inputs
    ----------
    file: list of sinkdata objects
    The initial data file
    
    systems:list of starsystem objects
    The data file made into systems.
    
    Parameters
    ----------
    upper_limit: int,float,opt
    The upper limit of the mass range.

    lower_limit: int,float,opt
    The lower limit of the mass range.
    
    target_mass:string,int,opt
    The target mass to print out on the plot.

    filename: string,opt
    The filename to print out on the plot.
    
    plot:bool,opt
    Whether to plot or return the data.
    
    min_time_bin:float,opt
    Miminum time to plot in each bin
    
    only_primaries_and_singles: bool,opt
    Whether to include all stars or ignore companions
    
    full_form_times:bool,opt
    Whether to return all the formation times of all stars

    Returns
    -------
    times: list
    The list of times in the simulation.
    
    new_stars_co:list
    The number of new stars in each bin.

    Example
    -------
    formation_time_histogram(M2e4_C_M_2e7,M2e4_C_M_2e7_systems,only_primaries_and_singles = True)
    '''
    if label is None: label = path.basename(filename)
    if target_mass is None:
        target_mass = (upper_limit+lower_limit)/2
    birth_times = []
    if only_primaries_and_singles is True:
        if systems is None:
            print('Please provide systems')
            return
        for i in range(len(systems[-1])):
            this_mass = systems[-1][i].primary
            if lower_limit<=this_mass<=upper_limit:
                birth_time = systems[-1][i].formation_time_Myr[systems[-1][i].m==systems[-1][i].primary][0]
                birth_times.append(birth_time)
    else:
        for i in range(len(file[-1].m)):
            this_id = file[-1].id[i]
            this_mass = file[-1].m[i]
            if lower_limit<=this_mass<=upper_limit:
                birth_time = file[-1].extra_data['ProtoStellarAge'][file[-1].id == this_id][0]*code_time_to_Myr
                birth_times.append(birth_time)
    times = time_array(file,'Myr')
    birth_times = np.array(birth_times)
    times,new_stars_co = hist(birth_times,bins = np.linspace(min(times),max(times),num = int((max(times)-min(times))/min_time_bin)))
    times = np.array(times)
    new_stars_co = np.insert(new_stars_co,0,0)
    if full_form_times is True:
        return birth_times
    if plot == True:
        plt.step(times,new_stars_co)
        if filename is not None:
            plt.text(max(times)/2,max(new_stars_co),label)
        plt.text(max(times)/2,max(new_stars_co)*0.9,'Star Mass = '+str(target_mass)+' $\mathrm{\mathrm{M_\odot}}$')
        plt.xlabel('Time [Myr]')
        plt.ylabel('Number of New Stars')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
        plt.figure(figsize = (6,6))
    else:
        return times,new_stars_co

def formation_density_histogram(file,systems,upper_limit=1.3,lower_limit = 0.7,target_mass = None,filename = None,plot = True,min_dens_bin = 0.2,only_primaries_and_singles = False,full_form_dens = False,density = 'number',label=None):
    '''
    Create or return a histogram of the formation times of stars in the given mass range.
    
    Inputs
    ----------
    file: list of sinkdata objects
    The initial data file
    
    systems:list of starsystem objects
    The data file made into systems.
    
    Parameters
    ----------
    upper_limit: int,float,opt
    The upper limit of the mass range.

    lower_limit: int,float,opt
    The lower limit of the mass range.
    
    target_mass:string,int,opt
    The target mass to print out on the plot.

    filename: string,opt
    The filename to print out on the plot.
    
    plot:bool,opt
    Whether to plot or return the data.
    
    min_dens_bin:float,opt
    Miminum density to plot in each bin
    
    only_primaries_and_singles: bool,opt
    Whether to include all stars or ignore companions
    
    full_form_dens:bool,opt
    Whether to return all the formation densities of all stars
    
    density: string,opt
    Whether to return 'mass' or 'number' density

    Returns
    -------
    times: list
    The list of times in the simulation.
    
    new_stars_co:list
    The number of new stars in each bin.

    Example
    -------
    formation_time_histogram(M2e4_C_M_2e7,M2e4_C_M_2e7_systems,only_primaries_and_singles = True)
    '''
    if label is None: label = filename
    if target_mass is None:
        target_mass = (upper_limit+lower_limit)/2
    birth_densities = []
    if only_primaries_and_singles is True:
        if systems is None:
            print('Please provide systems')
            return
        for i in range(len(systems[-1])):
            this_mass = systems[-1][i].primary
            if lower_limit<=this_mass<=upper_limit:
                birth_densities.append( np.log10(systems[-1][i].init_density[density])[0] )
    else:
        for i in range(len(file[-1].m)):
            this_id = file[-1].id[i]
            this_mass = file[-1].m[i]
            if lower_limit<=this_mass<=upper_limit:
                birth_density = np.log10(initial_local_density(this_id,file,density = density)[0])
                birth_densities.append(birth_density)
    birth_densities = np.array(birth_densities)
    densities,new_stars_co = hist(birth_densities,bins = np.linspace(0,max(birth_densities),num = int((max(birth_densities))/min_dens_bin)))
    densities = np.array(densities)
    new_stars_co = np.insert(new_stars_co,0,0)
    if full_form_dens is True:
        return birth_densities
    if plot == True:
        plt.step(densities,new_stars_co)
        if label is not None:
            plt.text(max(densities)/2,max(new_stars_co),label)
        plt.text(max(densities)/2,max(new_stars_co)*0.9,'Star Mass = '+str(target_mass)+' $\mathrm{\mathrm{M_\odot}}$')
        if density == 'number':
            plt.xlabel(r'Density [$\mathrm{pc}^{-3}$]')
        else:
            plt.xlabel(r'Density [$\frac{\mathrm{M_\odot}}{pc^{3}}$]')
        plt.ylabel('Number of New Stars')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
        plt.figure(figsize = (6,6))
    else:
        return densities,new_stars_co

def star_formation_rate(file,plot = True,time = True,filename = None,time_norm = True,rolling_avg = False,rolling_window_Myr = 0.1):
    '''
    Average star formation rate[dM/dt] (at every snapshot in Myr) of all stars
    
    Inputs
    ----------
    file : list of sinkdata.
    The input data that will be grouped into systems.

    Parameters
    ----------
    plot : bool,optional
    Whether to plot the number of stars.

    time: bool,optional
    Whether to have snapshot number or time as the x axis.

    time_norm: bool,optional
    Whether to normalize the time by ff_time or not

    rolling_avg: bool,optional
    Whether to include a rolling average instead of every data point.
    
    rolling_window_Myr: int,float,optional
    The time to include into the rolling average window. [in Myr]

    Returns
    -------
    SFR: list
    The SFR at each time.

    Example
    -------
    1) star_formation_rate(M2e4_C_M_2e7,time = True)
    Plotting the SFR over time.

    2) star_formation_rate(M2e4_C_M_2e7,time = True,rolling_avg = True,rolling_window_Myr = 10)
    Plotting the star formation rate over a rolling window.
    '''
    time_step = (file[-1].t - file[-2].t)*code_time_to_Myr
    SFR = []
    times = []
    rolling_window = int((int(time_to_snaps(rolling_window_Myr,file))//2)*2+1)
    for i in range(1,len(file)):
        current_time = file[i].t*code_time_to_Myr
        current_mass = sum(file[i].m)
        previous_time = file[i-1].t*code_time_to_Myr
        previous_mass = sum(file[i-1].m)
        dm = current_mass - previous_mass
        dt = current_time - previous_time
        SFR.append(dm/dt)
        if time_norm == True:
            ff_t = t_ff(file_properties(filename,param = 'm'),file_properties(filename,param = 'r'))
            time = (file[i].t/(ff_t*np.sqrt(file_properties(filename,param = 'alpha'))))
            times.append(time)
        else:
            times.append(current_time)
    if plot ==True:
        plt.figure(figsize = (6,6))
        if rolling_avg is True:
            plt.plot(times,rolling_average(SFR,rolling_window = rolling_window))
        else:
            plt.plot(times,SFR)
        if time_norm is False:
            plt.xlabel('Time [Myrs]')
        else:
            plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
        plt.ylabel(r'Star Formation Rate [$\frac{\mathrm{M_\odot}}{Myr}$]')
        if filename is not None:
            if filename != 'M2e4_C_M_J_RT_W_R30_2e7':
                plt.text(times[-1]*0.5,max(SFR)*0.9,filename)
            else:
                plt.text(1.4,max(SFR)*0.9,filename)
            
        plt.yscale('log')
        adjust_font(ax_fontsize=14,labelfontsize=14)
    else:
        return SFR

def average_star_age(file,plot = True,time = True,rolling_avg = True,rolling_window_Myr = 0.1):
    '''Average age (at every snapshot in Myr) of all stars'''
    average_ages = []
    times = []
    rolling_window = int((int(time_to_snaps(rolling_window_Myr,file))//2)*2+1)
    for i in file:
        current_time = i.t*code_time_to_Myr
        ages = copy.copy(i.formation_time)
        ages = (current_time - ages*code_time_to_Myr)
        average_age = np.average(ages)
        average_ages.append(average_age)
        times.append(i.t*code_time_to_Myr)
    if plot ==True:
        if rolling_avg is False:
            plt.plot(times,average_ages)
        else:
            plt.plot(rolling_average(times,rolling_window),rolling_average(average_ages,rolling_window))
    else:
        return average_ages

def slope_to_mean_lim(slope,limits,num=50):
    'Converts slope to mean'
    x=np.linspace(limits[0],limits[1],num=num)
    return np.trapz(x**(slope+1),x=x)/np.trapz(x**(slope),x=x)
    
def diff_mean_lim(slope,target,limits):
    'Slope to mean mass - Target'
    return slope_to_mean_lim(slope,limits)-target

def mean_lim_to_slope(mass,limits):
    'Converts a mean mass into the IMF slope'
    slope_est = optimize.root(diff_mean_lim, -2.3, args=(mass,limits))['x'][0] 
    return np.clip(slope_est,-6,0)

def primary_stars_slope(file,systems,snapshot,lower_limit = 1,upper_limit = 10,slope = True,no_of_stars = False):
    '''
    Finds the slope (or mean mass) of stars with mass within the lower and upper limit for a single snapshot.
    
    Inputs
    ----------
    file : list of sinkdatas.
    The input data that isn't grouped into systems.
    
    systems: list of star systems
    The input data that is grouped into systems.
    
    snapshot: int
    The snapshot to look at

    Parameters
    ----------
    lower_limit: int,float,optional
    The lower limit of the mass range

    upper_limit: int,float,optional
    The upper limit of the mass range
    
    slope: bool,optional
    Whether to return the slope or the mean mass
    
    no_of_stars:bool,optional
    Whether to return the no of stars (that are primaries and the total number also)

    Returns
    -------
    slope_all: int
    The slope of all stars within the mass range.
    
    slope_prim:int
    The slope of all primary stars within the mass range
    
    no_all:int
    The number of stars withing the mass range.
    
    no_prim:int
    The number of primaries within the mass range.

    Example
    -------
    1) primary_stars_slope(M2e4_C_M_2e7,M2e4_C_M_2e7_systems,-1,upper_limit = 10,lower_limit = 1)
    Returns the slopes of all the stars and the primaries in the mass range of [1,10].

    2) primary_stars_slope(M2e4_C_M_2e7,M2e4_C_M_2e7_systems,-1,upper_limit = 10,lower_limit = 1,slope = False,no_of_stars = True)
    Returns the average mass of all the stars and the primaries in the mass range of [1,10], as well as their counts.
    
    '''
    total_mass_all = 0
    no_all = 0
    for i in file[snapshot].m:
        if lower_limit<=i<=upper_limit:
            total_mass_all += i
            no_all+=1
    if no_all>0:
        avg_all = total_mass_all/no_all
    else:
        avg_all = np.nan
    total_mass_prim = 0
    no_prim = 0
    for i in systems[snapshot]:
        if lower_limit<=i.primary<=upper_limit:
            total_mass_prim += i.primary
            no_prim+=1
    if no_prim>0:
        avg_prim = total_mass_prim/no_prim
    else:
        avg_prim = np.nan
    if slope is True:
        if avg_all is np.nan and avg_prim is np.nan:
            if no_of_stars is False:
                return np.nan,np.nan
            else:
                return np.nan,np.nan,np.nan,np.nan
        elif avg_all is np.nan:
            if no_of_stars is False:
                return np.nan,avg_prim
            else:
                return np.nan,avg_prim,np.nan,no_prim
        elif avg_prim is np.nan:
            if no_of_stars is False:
                return avg_all,np.nan
            else:
                return avg_all,np.nan,no_all,np.nan
        slope_all = mean_lim_to_slope(avg_all,[lower_limit,upper_limit])
        slope_prim = mean_lim_to_slope(avg_prim,[lower_limit,upper_limit])
        if no_of_stars is True:
            return slope_all,slope_prim,no_all,no_prim
        else:
            return slope_all,slope_prim

def slope_evolution(file,systems,filename,lower_limit = 1,upper_limit = 10,no_of_stars = False,min_no = 1,plot = True,rolling_avg = False,rolling_window_Myr = 0.1):
    '''
    Tracks the slope of stars with mass within the lower and upper limit throughout the simulation runtime.
    
    Inputs
    ----------
    file : list of sinkdata.
    The input data that isn't grouped into systems.
    
    systems: list of star systems
    The input data that is grouped into systems.
    
    filename: string
    The name of the file.

    Parameters
    ----------
    lower_limit: int,float,optional
    The lower limit of the mass range

    upper_limit: int,float,optional
    The upper limit of the mass range
    
    no_of_stars:bool,optional
    Whether to return the no of stars (that are primaries and the total number also) plot also
    
    min_no: int,optional
    The minimum number of stars at the start of the plot time.
    
    plot: bool,optional
    Whether to plot the quantities or return the slopes lists.
    
    rolling_avg:bool,optional
    Whether to use a rolling average or not.
    
    rolling_window_Myr:int,float,optional
    Time to use in the rolling average. [in Myr]

    Returns
    -------
    slope_all: int
    The slope of all stars within the mass range.
    
    slope_prim:int
    The slope of all primary stars within the mass range
    
    no_all:int
    The number of stars withing the mass range.
    
    no_prim:int
    The number of primaries within the mass range.

    Example
    -------
    1) primary_stars_slope(M2e4_C_M_2e7,M2e4_C_M_2e7_systems,-1,upper_limit = 10,lower_limit = 1)
    Returns the slopes of all the stars and the primaries in the mass range of [1,10].

    2) primary_stars_slope(M2e4_C_M_2e7,M2e4_C_M_2e7_systems,-1,upper_limit = 10,lower_limit = 1,slope = False,no_of_stars = True)
    Returns the average mass of all the stars and the primaries in the mass range of [1,10], as well as their counts.
    
    '''
    all_stars_slopes = []
    primary_stars_slopes = []
    times = []
    nos_prim = []
    nos_all = []
    rolling_window = int((int(time_to_snaps(rolling_window_Myr,file))//2)*2+1)
    for i in range(len(file)):
        slope_all,slope_prim,no_all,no_prim = primary_stars_slope(file,systems,i,lower_limit=lower_limit,upper_limit=upper_limit,slope = True,no_of_stars=True)
        nos_prim.append(no_prim)
        nos_all.append(no_all)
        all_stars_slopes.append(slope_all)
        primary_stars_slopes.append(slope_prim)
        ff_t = t_ff(file_properties(filename,param = 'm'),file_properties(filename,param = 'r'))
        time = (file[i].t/(ff_t*np.sqrt(file_properties(filename,param = 'alpha'))))
        times.append(time)
    all_stars_slopes = np.array(all_stars_slopes);primary_stars_slopes = np.array(primary_stars_slopes)
    nos_prim = np.array(nos_prim);nos_all = np.array(nos_all)
    times = np.array(times)
    if plot is True:
        if rolling_avg is True:
            #times = np.array(rolling_average(times,rolling_window))
            all_stars_slopes = np.array(rolling_average(all_stars_slopes,rolling_window))
            primary_stars_slopes = np.array(rolling_average(primary_stars_slopes,rolling_window))
            nos_all = np.array(rolling_average(nos_all,rolling_window))
            nos_prim = np.array(rolling_average(nos_prim,rolling_window))
        plt.figure(figsize = (6,6))
        plt.plot(times[nos_all>min_no],all_stars_slopes[nos_all>min_no],label = 'Slope for All')
        plt.plot(times[nos_all>min_no],primary_stars_slopes[nos_all>min_no],label = 'Slope for Primary',linestyle = '--')
        plt.plot(times[nos_all>min_no],[-2.3]*len(times[nos_all>min_no]),label = '-2.3',linestyle = ':')
        plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
        plt.ylabel('Slope of stars in Mass Range')
        plt.text((times[-1]+times[nos_all>min_no][0])/2,primary_stars_slopes[-1],filename)
        plt.legend(fontsize = 14)
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
        plt.show()
        if no_of_stars is True:
            plt.figure(figsize = (6,6))
            plt.plot(times[nos_all>min_no],nos_all[nos_all>min_no],label = 'Number for All')
            plt.plot(times[nos_all>min_no],nos_prim[nos_all>min_no],label = 'Number for Primary',linestyle = '--')
            plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
            plt.ylabel('Number of Stars in Mass Range')
            plt.legend(fontsize = 14)
            plt.text((times[-1]+times[nos_all>min_no][0])/2,nos_all[-1],filename)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
            plt.show()
    else:
        return all_stars_slopes,primary_stars_slopes

def percentile_75(array):
    return np.percentile(array,75)

def percentile_25(array):
    return np.percentile(array,25)

def density_evolution(densities,times,bins = 10,plot = True,filename = None,density = 'number'):
    '''
    A plot of the mean local density throughout formation times in the simulation
    
    Inputs
    ----------
    densities : list,array
    The list of densities (calculated from the inital density function)
    
    times: list,array
    The list of formation times of stars.

    Parameters
    ----------
    bins: int,string,list
    The bins used.
    
    plot: bool,optional
    Whether to plot the quantities or return the times,mean densities and the standard deviations.
    
    filename: string
    The name of the file.
    
    density:string
    Either mass density or number density

    Returns
    -------
    binned_times: array
    The formation times.
    
    means:int
    The mean local density in each bin.
    
    stds:int
    The standard deviation in each bin.

    Example
    -------
    1) density_evolution(densities,times,'M2e4_C_M_2e7')
    Plots the average formation local density over time.

    2) density_evolution(densities,times,'M2e4_C_M_2e7',plot = False)
    Returns the times, the average formation local density over time and the standard deviation of local density over time.
    '''
    
    if not hasattr(bins, "__iter__"):
        edges = advanced_binning(times,np.ptp(times)/bins, min_bin_pop=5,allow_empty_bins=False)
    else:
        edges = bins

    means,binned_times,bindices = stats.binned_statistic(times,densities,statistic='median',bins = edges)
    val75,binned_times,bindices = stats.binned_statistic(times,densities,statistic=percentile_75,bins = edges)
    val25,binned_times,bindices = stats.binned_statistic(times,densities,statistic=percentile_25,bins = edges)
    count_per_bin,binned_times,bindices = stats.binned_statistic(times,densities,statistic='count',bins = edges)
    error_up = val75-means; error_down=means-val25
    
    if plot == True:
        plt.plot((binned_times[1:]+binned_times[:-1])/2,means,marker = 'o',color = 'indianred')
        plt.fill_between((binned_times[1:]+binned_times[:-1])/2,val75,val25,alpha = 0.2,color = 'indianred')
        plt.xlabel('Formation Time [Myr]')
        if density == 'number':
            plt.ylabel(r'Log Local Stellar Density [$\mathrm{pc}^{-3}$]')
        else:
            plt.ylabel(r'Log Local Stellar Mass Density [$\mathrm{M_\odot}pc^{-3}$]')
        # if filename is not None:
        #     plt.text(max(binned_times)*0.5,np.log10(max(means))*0.9,filename)
        adjust_font(fig = plt.gcf())
        #plt.yscale('log')
        #print(count_per_bin)
    else:
        return binned_times,means,error_up,error_down

def momentum_angle(id1,id2,file,snapshot):
    if id1 == id2:
        return 0
    else:
        momentum1 = file[snapshot].extra_data['BH_Specific_AngMom'][file[snapshot].id == id1][0]
        momentum2 = file[snapshot].extra_data['BH_Specific_AngMom'][file[snapshot].id == id2][0]
        dot_product = np.dot(momentum1,momentum2)
        normal1 = np.linalg.norm(momentum1);normal2 = np.linalg.norm(momentum2)
        cosangle = dot_product/(normal1*normal2)
        angle = np.arccos(cosangle)
        return angle*180/np.pi

def Seperation_Tracking(file,systems,rolling_avg = False):
    filtered_systems = get_q_and_time(systems[-1],systems)
    tracking_systems = []
    for j in filtered_systems:
        if j.no == 2 and j.primary>0.08:
            tracking_systems.append(j)
    x_axis = []
    y_axis = []
    initial_sep = []
    for i in tqdm(tracking_systems,position=0):
        smaxes,times = distance_tracker_binaries(file,systems,i.ids,plot = False,rolling_avg=False)
        x_axis.append(times)
        y_axis.append(smaxes)
        initial_sep.append(np.array(smaxes)[np.array(times) == 0])
    seperations = np.linspace(2,6,5)
    seperation_random = np.insert(seperations,0,0)
    x_axis_sep = []
    y_axis_sep = []
    for i in range(len(seperations)):
        x_axis_sep.append([])
        y_axis_sep.append([])
    for i in range(len(initial_sep)):
        for j in range(len(seperations)):
            if seperation_random[j]<=initial_sep[i][0]<=seperation_random[j+1]:
                x_axis_sep[j].append(x_axis[i])
                y_axis_sep[j].append(y_axis[i])
    for i in tqdm(range(len(x_axis_sep))):
        plt.figure(figsize = (6,6))
        rolling_window = time_to_snaps(0.1,file)
        if rolling_window%2 == 0:
            rolling_window -= 1
        rolling_window = int(rolling_window)
        maximum = 0
        for j in range(len(x_axis_sep[i])):
            if rolling_avg is True:
                #x_axis_rolled = rolling_average(x_axis_sep[i][j])
                x_axis_rolled = x_axis_sep[i][j]
                y_axis_rolled = rolling_average(y_axis_sep[i][j])
                plt.plot(x_axis_rolled,y_axis_rolled)
                if np.max(y_axis_rolled) > maximum:
                    maximum = np.max(y_axis_rolled)
            else:
                plt.plot(x_axis_sep[i][j],y_axis_sep[i][j])
                if np.max(y_axis_sep[i][j])>maximum:
                    maximum = np.max(y_axis_sep[i][j])
        plt.xlabel('Age [Myr]')
        plt.ylabel('Log Seperation [AU]')
        plt.text(0,maximum,'Log '+str(seperation_random[i].round(2))+' AU''< Initial Seperation < Log '+str(seperation_random[i+1].round(2))+' AU')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.savefig('Upto '+str(seperation_random[i+1].round(2))+'.png',dpi = 100,bbox_inches = 'tight')

#Getting the total masses, primary masses, smaxes and companion mass ratio. Also gets the target primary masses for 
#smaxes and companion mass ratios.
def primary_total_ratio_axis(systems,lower_limit = 0,upper_limit = 10000,all_companions = False,attribute = 'Mass Ratio',file = False, age_limit_Myr=1e10):
    '''
    Returns a list of the property you chose for systems with primaries in a certain mass range.

    Inputs
    ----------
    systems : list of star system objects.
    The systems in a certain snapshot to be looked at.

    Parameters
    ----------
    lower limit : int,float,optional
    The lower limit of the primary mass range.

    upper limit : int,float,optional
    The upper limit of the primary mass range.
        
    all_companions: bool,optional
    Whether to include all companions or just the most massive (for mass ratio) or all subsystems or just the subsystems with the primary and secondary (Semi Major Axis)

    attribute: string,optional
    The attribute that you want. Choose from 'System Mass','Primary Mass','Mass Ratio', 'Semi Major Axis' or 'Angle'.
    
    file: list of star system objects
    The file that the systems are from. Only temp use for angles
    
    age_limit_Myr: float, optional
    If set only systems younger than this value are considered

    Returns
    -------
    attribute_distribution: list
    The distribution of the property that you requested.

    Example
    -------
    1) primary_total_ratio_axis(M2e4_C_M_2e7_systems[-1],attribute = 'System Mass')
    Returns the mass of all multi star systems 

    2) primary_total_ratio_axis(M2e4_C_M_2e7_systems[-1],attribute = 'Primary Mass')
    Returns the mass of all primaries in multi star systems 

    3) primary_total_ratio_axis(M2e4_C_M_2e7_systems[-1],attribute = 'Mass Ratio',all_companions = True,lower_limit = 0.7,upper_limit = 1.3)
    Returns the mass ratio of all solar type companions.

    1) primary_total_ratio_axis(M2e4_C_M_2e7_systems[-1],attribute = 'Semi Major Axis',all_companions = True,lower_limit = 0.7,upper_limit = 1.3)
    Returns the semi major axis of all subsystems in the system.

    '''

    masses = []
    primary_masses = []
    mass_ratios = []
    semi_major_axes = []
    angles = []
    eccentricities = []
    for i in systems:
        if len(i.m)>1: #Make sure you only consider the multi star systems.
            if (i.age_Myr[np.argmax(i.m)]<=age_limit_Myr):
                masses.append(i.tot_m)
                primary_masses.append(i.primary)
                if lower_limit<=i.primary<=upper_limit:
                    if all_companions == False:
                        semi_major_axes.append(i.smaxis)
                        mass_ratios.append(i.mass_ratio)
                        snapshot = i.snapshot_num
                        secondary_id = i.ids[i.m == i.secondary][0]
                        angles.append(momentum_angle(i.primary_id,secondary_id,file,snapshot))
                        eccentricities.append(ecc_smaxis_in_system(i.primary_id,secondary_id,i))
                    elif all_companions == True: #If you want to look at all companions or subsystems.
                        semi_major_axes = semi_major_axes +list(i.smaxis_all)
                        eccentricities = eccentricities + list(i.ecc_all)
                        snapshot = i.snapshot_num
                        snapshot = -1
                        for j in i.m:
                            if j!= i.primary:
                                mass_ratios.append(j/i.primary)
                                secondary_id = i.ids[i.m == j][0]
                                angles.append(momentum_angle(i.primary_id,secondary_id,file,snapshot))
                            
    if attribute == 'System Mass':
        return masses
    elif attribute == 'Primary Mass':
        return primary_masses
    elif attribute == 'Mass Ratio':
        return mass_ratios
    elif attribute == 'Semi Major Axis':
        return list(flatten(semi_major_axes))
    elif attribute == 'Angle':
        return angles
    elif attribute == 'Eccentricity':
        return eccentricities
    else:
        return None

#Multiplicity Fraction over different masses with a selection ratio of companions
def multiplicity_fraction(systems,mass_break = 2,selection_ratio = 0,attribute = 'Fraction',bins = 'continous'):
    '''
    Returns the multiplicity fraction or multiplicity properties over a mass range.

    Inputs
    ----------
    systems : list of star system objects.
    The systems in a certain snapshot to be looked at.

    Parameters
    ----------
    mass_break : int,float,optional
    The log seperation in masses.

    selection_ratio : int,float,optional
    The minimum mass ratio of the companions.
        
    attribute: string,optional
    The attribute that you want. Choose from 'Fraction'(Primary No/(Primary No+ Single No)),'All Companions'(Primary No/(Primary No+Single No+Companion Number) or 'Properties'.

    bins: string,optional
    The type of bins that you want. Choose from 'continous' (evenly spaced in log space) or 'observer' (Duchene Krauss bins).

    Returns
    -------
    logmasslist: list
    The list of masses in logspace.

    Multiplicity_Fraction_List or Single & Primary & Companion_Fractions_List: list
    The list of multiplicity fraction or the 3 lists of primary,single or companion fractions.

    mult_sys_count: int
    The number of systems with more than one star (When returning multiplicity fraction).

    sys_count:int
    The number of systems (including single star systems)(When returning multiplicity fraction).

    Example
    -------
    1) multiplicity_fraction(M2e4_C_M_2e7_systems[-1],attribute = 'Fraction',bins = 'observer') 
    Returns the logmasslist, multiplicity fraction list, count of all multiple star systems and the number of all systems with the Duchene Krauss Bins.

    2) multiplicity_fraction(M2e4_C_M_2e7_systems[-1],attribute = 'Properties',bins = 'continous') 
    Returns the logmasslist, single star fraction list, primary star fraction list and the companion star fraction list.
    '''

    m = []
    state = []
    for i in systems:
        if len(i.m) == 1:
            m.append(i.m[0])
            state.append(0)
        elif i.no>1:
            for j in i.m:
                if j>=i.primary*selection_ratio and j != i.primary:
                    m.append(j)
                    state.append(-1)
            m.append(i.primary)
            masses = np.array(i.m)
            if len(masses[masses>=selection_ratio*i.primary])>1:
                state.append(1)
            elif len(masses[masses>=selection_ratio*i.primary]) == 1:
                state.append(0)

    minmass= 0.08 # Because we dont want brown dwarfs
    maxmass = max(m)
    if bins == 'continous':
        logmasslist= np.linspace(np.log10(minmass),np.log10(maxmass+1),num = int((np.log10(maxmass+1)-np.log10(minmass))/(np.log10(mass_break))))
    elif bins == 'observer':
        #masslist = np.array([0.1,0.7,1.5,5,8,16,maxmass+1])
        masslist = np.array([0.08,0.1, 0.15,0.3,0.7,1.3,2,8,16 ])
        if maxmass>16:
            masslist = np.append(masslist, maxmass+1)
        logmasslist = np.log10(masslist)
    log_m = np.log10(m)
    state = np.array(state)
    solo_count,_, _ = stats.binned_statistic(log_m,(state==0),statistic='sum',bins = logmasslist)
    primary_count,_, _ = stats.binned_statistic(log_m,(state==1),statistic='sum',bins = logmasslist)
    nonprimary_count,_, _ = stats.binned_statistic(log_m,(state==-1),statistic='sum',bins = logmasslist)  
    sys_count = primary_count+solo_count
    MF = primary_count/sys_count
    sys_count_alt = primary_count+solo_count+nonprimary_count
    MF_alt = primary_count/sys_count_alt
    logmass_bins = (logmasslist[1:]+logmasslist[:-1])/2

    if attribute == 'MF':
        return logmass_bins,MF,primary_count,sys_count
    elif attribute == 'All Companions':
        return logmass_bins,MF_alt,primary_count,sys_count_alt
    elif attribute == 'Properties':
        return logmass_bins,solo_count/sys_count_alt,primary_count/sys_count_alt,nonprimary_count/sys_count_alt
    else:
        return None

#Multiplicity Fraction over different masses with a selection ratio of companions
def multiplicity_fraction_with_density(systems,file,mass_break = 2,selection_ratio = 0,attribute = 'MF',bins = 'continous'):
    '''
    Returns the multiplicity fraction or multiplicity properties over a mass range or the density.

    Inputs
    ----------
    systems : list of star system objects.
    The systems in a certain snapshot to be looked at.

    file: list of sinkdata objects.
    The file before system assignment to be looked at.

    Parameters
    ----------
    mass_break : int,float,optional
    The log seperation in masses.

    selection_ratio : int,float,optional
    The minimum mass ratio of the companions.

    attribute: string,optional
    The attribute that you want. Choose from 'Fraction'(Primary No/(Primary No+ Single No)),'All Companions'(Primary No/(Primary No+Single No+Companion Number), 'Properties','Initial Densities','Initial Mass Densities','Initial Densities Separate','Initial Mass Densities Separate'.

    bins: string,optional
    The type of bins that you want. Choose from 'continous' (evenly spaced in log space) or 'observer' (Duchene Krauss bins).

    Returns
    -------
    logmasslist: list
    The list of masses in logspace.

    Multiplicity_Fraction_List or Single & Primary & Companion_Fractions_List or Densities List(s): list
    The list of multiplicity fraction or the 3 lists of single,primary or companion fractions.

    mult_sys_count: int
    The number of systems with more than one star (When returning multiplicity fraction).

    sys_count:int
    The number of systems (including single star systems)(When returning multiplicity fraction).

    Example
    -------
    1) multiplicity_fraction(M2e4_C_M_2e7_systems[-1],attribute = 'Fraction',bins = 'observer') 
    Returns the logmasslist, multiplicity fraction list, count of all multiple star systems and the number of all systems with the Duchene Krauss Bins.

    2) multiplicity_fraction(M2e4_C_M_2e7_systems[-1],attribute = 'Properties',bins = 'continous') 
    Returns the logmasslist, single star fraction list, primary star fraction list and the companion star fraction list.

    3) multiplicity_fraction(M2e4_C_M_2e7_systems[-1],attribute = 'Density',bins = 'observer') 
    Returns the logmasslist, formation density list, count of all multiple star systems and the number of all systems with the Duchene Krauss Bins.

    4) multiplicity_fraction(M2e4_C_M_2e7_systems[-1],attribute = 'Mass Density Seperate',bins = 'observer') 
    Returns the logmasslist and the formation mass density lists of solo stars, primary stars and companion stars.
    '''

    m = []
    state = []
    mass_densities = []
    number_densities = []
    for i in tqdm(systems,position = 0,desc = 'processing all systems'):
        selection_index = (i.m>=i.primary*selection_ratio)
        mass_densities.append(i.init_star_mass_density[selection_index])
        number_densities.append(i.init_star_vol_density[selection_index])
        state.append(i.multip_state[selection_index])
        m.append(i.m)
    mass_densities = np.concatenate(mass_densities)
    number_densities = np.concatenate(number_densities)
    state = np.concatenate(state)
    m = np.concatenate(m)

    minmass= 0.1 # Because we dont want brown dwarfs
    maxmass = max(m)
    if bins == 'continous':
        logmasslist= np.linspace(np.log10(minmass),np.log10(maxmass+1),num = int((np.log10(maxmass+1)-np.log10(minmass))/(np.log10(mass_break))))
    elif bins == 'observer':
        #masslist = np.array([0.1,0.7,1.5,5,8,16,maxmass+1])
        masslist = np.array([0.08,0.1,0.7,1.5,5,16,maxmass+1])
        if maxmass<16:
            masslist = np.array([0.08,0.1,0.7,1.5,5,16])
        logmasslist = np.log10(masslist)
    primary_fraction = np.zeros_like(logmasslist)
    single_fraction = np.zeros_like(logmasslist)
    secondary_fraction = np.zeros_like(logmasslist)
    other_fraction = np.zeros_like(logmasslist)
    alternative_fraction = np.zeros_like(logmasslist)
    sys_count = np.zeros_like(logmasslist)
    mult_sys_count = np.zeros_like(logmasslist)
    primary_densities = np.zeros_like(logmasslist)
    primary_mass_densities = np.zeros_like(logmasslist)
    secondary_densities = np.zeros_like(logmasslist)
    secondary_mass_densities = np.zeros_like(logmasslist)
    solo_densities = np.zeros_like(logmasslist)
    solo_mass_densities = np.zeros_like(logmasslist)
    other_densities = np.zeros_like(logmasslist)
    other_mass_densities = np.zeros_like(logmasslist)
    dens_errors = np.zeros_like(logmasslist)
    mass_dens_errors = np.zeros_like(logmasslist)
    dens_prim_errors = np.zeros_like(logmasslist)
    mass_dens_prim_errors = np.zeros_like(logmasslist)
    dens_sec_errors = np.zeros_like(logmasslist)
    mass_dens_sec_errors = np.zeros_like(logmasslist)
    dens_solo_errors = np.zeros_like(logmasslist)
    mass_dens_solo_errors = np.zeros_like(logmasslist)
    ind = np.digitize(np.log10(m),logmasslist)
    bins = [[]]*len(logmasslist)
    mass_dens_bins = [[]]*len(logmasslist)
    dens_bins = [[]]*len(logmasslist)
    for i in range(len(bins)):
        bins[i] = []
        mass_dens_bins[i] = []
        dens_bins[i] = []
    for i in range(len(m)):
        bin_no = ind[i]-1 
        bins[bin_no].append(state[i])
        mass_dens_bins[bin_no].append(mass_densities[i])
        dens_bins[bin_no].append(number_densities[i])
    for i in range(len(bins)):
        primary_count = 0
        secondary_count = 0
        solo_count = 0
        primary_dens = 0
        secondary_dens = 0
        solo_dens = 0
        primary_mdens = 0
        secondary_mdens = 0
        solo_mdens = 0
        #dens_errors[i] = np.std(dens_bins[i])
        #mass_dens_errors[i] = np.std(mass_dens_bins[i])
        all_dens_error = []
        all_mdens_error = []
        prim_dens_error = []
        prim_mdens_error = []
        sec_dens_error = []
        sec_mdens_error = []
        solo_dens_error = []
        solo_mdens_error = []
        for j in range(len(bins[i])):
            if bins[i][j]==0:
                solo_count = solo_count + 1
                solo_dens += dens_bins[i][j]
                solo_mdens += mass_dens_bins[i][j]
                solo_dens_error.append(dens_bins[i][j])
                solo_mdens_error.append(mass_dens_bins[i][j])
                all_dens_error.append(dens_bins[i][j])
                all_mdens_error.append(mass_dens_bins[i][j])
            elif bins[i][j] >= 1:
                primary_count = primary_count + 1
                primary_dens += dens_bins[i][j]
                primary_mdens += mass_dens_bins[i][j]
                prim_dens_error.append(dens_bins[i][j])
                prim_mdens_error.append(mass_dens_bins[i][j])
                all_dens_error.append(dens_bins[i][j])
                all_mdens_error.append(mass_dens_bins[i][j])
            else:
                secondary_count = secondary_count + 1
                secondary_dens += dens_bins[i][j]
                secondary_mdens += mass_dens_bins[i][j]
                sec_dens_error.append(dens_bins[i][j])
                sec_mdens_error.append(mass_dens_bins[i][j])
        dens_errors[i] = np.std(all_dens_error)
        mass_dens_errors[i] = np.std(all_mdens_error)
        dens_prim_errors[i] = np.std(prim_dens_error)
        mass_dens_prim_errors[i] = np.std(prim_mdens_error)
        dens_sec_errors[i] = np.std(sec_dens_error)
        mass_dens_sec_errors[i] = np.std(sec_mdens_error)
        dens_solo_errors[i] = np.std(solo_dens_error)
        mass_dens_solo_errors[i] = np.std(solo_mdens_error)
        if len(bins[i])>0:
            primary_fraction[i] = primary_count/len(bins[i])
            single_fraction[i] = solo_count/len(bins[i])
            secondary_fraction[i] = secondary_count/len(bins[i])
        else:
            primary_fraction[i] = np.nan
            single_fraction[i] = np.nan
            secondary_fraction[i] = np.nan
        if primary_count+solo_count>0:
            other_fraction [i] = primary_count/(primary_count+solo_count)
            mult_sys_count[i] = primary_count
            other_densities[i] = (primary_dens+solo_dens)/(primary_count+solo_count)
            other_mass_densities[i] = (primary_mdens+solo_mdens)/(primary_count+solo_count)
            sys_count[i] = primary_count+solo_count
        else:
            other_fraction[i] = np.nan
            mult_sys_count[i] = np.nan
            sys_count[i] = np.nan
        if primary_count+solo_count+secondary_count>0:
            alternative_fraction[i] = primary_count/(primary_count+solo_count+secondary_count)
        else:
            alternative_fraction[i] = np.nan
        if primary_count>0:
            primary_densities[i] = primary_dens/primary_count
            primary_mass_densities[i] = primary_mdens/primary_count
        else:
            primary_densities[i] = np.nan
            primary_mass_densities[i] = np.nan
        if secondary_count>0:
            secondary_densities[i] = secondary_dens/secondary_count
            secondary_mass_densities[i] = secondary_mdens/secondary_count
        else:
            secondary_densities[i] = np.nan
            secondary_mass_densities[i] = np.nan
        if solo_count>0:
            solo_densities[i] = solo_dens/solo_count
            solo_mass_densities[i] = solo_mdens/solo_count
        else:
            solo_densities[i] = np.nan
            solo_mass_densities[i] = np.nan
    if attribute == 'MF':
        return logmasslist,other_fraction,mult_sys_count,sys_count
    elif attribute == 'All Companions':
        return logmasslist,alternative_fraction
    elif attribute == 'Properties':
        return logmasslist,single_fraction,primary_fraction,secondary_fraction
    elif attribute == 'Density':
        return logmasslist,other_densities,mass_dens_errors
    elif attribute == 'Mass Density':
        return logmasslist,other_mass_densities,dens_errors
    elif attribute == 'Density Separate':
        return logmasslist,solo_densities,primary_densities,secondary_densities,dens_solo_errors,dens_prim_errors,dens_sec_errors,
    elif attribute == 'Mass Density Separate':
        return logmasslist,solo_mass_densities,primary_mass_densities,secondary_mass_densities,mass_dens_solo_errors,mass_dens_prim_errors,mass_dens_sec_errors
    else:
        return None

#Companion Frequency over different masses with a selection ratio
def companion_frequency(systems,mass_break = 2,selection_ratio = 0,attribute = 'CF',bins = 'continous'):
    '''
    Returns the companion frequency over a mass range.

    Inputs
    ----------
    systems : list of star system objects.
    The systems in a certain snapshot to be looked at.

    Parameters
    ----------
    mass_break : int,float,optional
    The log seperation in masses.

    selection_ratio : int,float,optional
    The minimum mass ratio of the companions.
        
    bins: string,optional
    The type of bins that you want. Choose from 'continous' (evenly spaced in log space) or 'observer' (Moe-DiStefano bins).

    Returns
    -------
    logmasslist: list
    The list of masses in logspace.

    companion_frequency: list
    The list of multiplicity frequencies.

    companion_count: int
    The number of companions.

    sys_count:int
    The number of systems (including single star systems).

    Example
    -------
    companion_frequency(M2e4_C_M_2e7_systems[-1],bins = 'observer') 
    Returns the logmasslist, Companion Frequency list, count of the number of companions and the number of all systems with the Moe DiStefano Bins.

    '''
    m = []
    companions = []
    for i in systems:
        throw = 0
        m.append(i.primary)
        if i.no>1:
            throw = len(np.array(i.m)[np.array(i.m)<=selection_ratio*i.primary])
        companions.append(i.no-1-throw)
    minmass= 0.08 # Because we dont want brown dwarfs
    maxmass= max(m)
    if bins == 'continous':
        logmasslist= np.linspace(np.log10(minmass),np.log10(maxmass),num = int((np.log10(maxmass)-np.log10(minmass))/(np.log10(mass_break))))
    elif bins == 'observer':
        #masslist = np.array([0.1,0.7,1.5,5,8,16,maxmass+1])
        masslist = np.array([0.08,0.1, 0.15,0.3,0.7,1.3,2,8,16 ])
        #if ( np.sum(np.array(m)>17) > 3 ):
        if maxmass>16:
            masslist = np.append(masslist, maxmass+1)
        logmasslist = np.log10(masslist)
    ind = np.digitize(np.log10(m),logmasslist)
    logmasslist = (logmasslist[1:]+logmasslist[:-1])/2 #we need bins not edges here
    bins = [[]]*len(logmasslist)
    for i in range(len(bins)):
        bins[i] = []
    for i in range(len(m)):
        bin_no = ind[i]-1 
        if (bin_no<len(bins)) and (bin_no>=0) :
            bins[bin_no].append(companions[i])
    companion_frequency = np.zeros_like(bins)
    companion_count = np.zeros_like(bins)
    sys_count = np.zeros_like(bins)
    for i in range(len(bins)):
        sys_count[i] = len(bins[i])
        companion_count[i] = sum(bins[i])
        if sys_count[i] == 0:
            companion_frequency[i] = np.nan
        else:
            companion_frequency[i] = sum(bins[i])/len(bins[i])
    return logmasslist,companion_frequency,companion_count,sys_count

#This is the weighted probability sum of the chances of having the number of companions. This allows us to check if companions
#are randomly distributed or not
def randomly_distributed_companions(systems,file,snapshot,lower_limit = 1/1.5,upper_limit = 1.5,target_mass = 1,mass_ratio = np.linspace(0,1,num = 11),plot = True):
    '''
    Returns the expected distribution of secondary companions if the drawing was random.

    Inputs
    ----------
    systems : list of star system objects.
    The systems in a certain snapshot to be looked at.

    file: list of sinkdata objects
    The original file before system assignment.

    snapshot: int
    The snapshot that you want to look at

    Parameters
    ----------
    lower_limit : int,float,optional
    The lower limit of the primary mass range.

    upper_limit : int,float,optional
    The upper limit of the primary mass range.

    target_mass : int,float,optional
    The target mass of primaries to cut the IMF at.
        
    mass_ratio: range,list,array,optional
    The bins for the mass ratios. Default is np.linspace(0,1,11)

    plot: bool,optional
    Whether to expected distribution or not

    Returns
    -------
    Nsystems_with_M_companion_mass: array
    The log number of companions expected at a certain mass ratio.

    Stellar_Mass_PDF: array
    The normalized log IMF until the primary mass

    Example
    -------
    1) randomly_distributed_companions(M2e4_C_M_2e7_systems[-1],M2e4_C_M_2e7,-1,attribute = 'Fraction') 
    Plots the expected companion distribution if the stars were randomly drawn from the IMF.
    '''
    systems_target = []
    count = 0
    for i in systems:
        if lower_limit<=i.primary<=upper_limit:
            systems_target.append(i)
        count += 1
    no2 = 0
    no3 = 0
    no4 = 0
    for i in systems_target:
        if i.no == 2:
            no2 = no2 +1
        elif i.no == 3:
            no3 = no3 + 1
        elif i.no == 4:
            no4 = no4 + 1
    if no2+no3+no4 == 0:
        w2 = np.nan;w3 = np.nan;w4 = np.nan
    else:
        w2 = no2/(no2+no3+no4)
        w3 = no3/(no2+no3+no4)
        w4 = no4/(no2+no3+no4)
    
    m_sorted = np.sort(file[snapshot].m[file[snapshot].m<(target_mass)])
    Nstars = len(m_sorted)
    P_full = np.arange(Nstars)/Nstars
    
    plot_masses = np.array(mass_ratio)*target_mass
    P_array = np.interp(plot_masses,m_sorted,P_full)
    
    Stellar_Mass_PDF = np.diff(P_array)*len(systems_target)
    
    P1 = w2* P_array + w3*(P_array)**2 + w4*(P_array)**3
    
    probability_M = np.diff(P1)
    
    Nsystems_with_M_companion_mass = probability_M*len(systems_target)
    
    Nsystems_with_M_companion_mass = np.insert(Nsystems_with_M_companion_mass,0,0)
    if plot == True:
        plt.step(mass_ratio,Nsystems_with_M_companion_mass)
        plt.yscale('log')
    else:
        return Nsystems_with_M_companion_mass,Stellar_Mass_PDF

#Describes the time evolution of the MF/CF of different masses with two lines, one that
#shows the multiplicity at a given time and one that only chooses stars that remain solar mass
def MFCF_Time_Evolution(file,Master_File,filename,steps=1,read_in_result = True,start = 0,target_mass = 1,upper_limit = 1.5,lower_limit = 1/1.5,plot = True,rolling_avg = True,rolling_window_Myr = 0.1,time_norm = 'tff',multiplicity = 'MF'):
    '''
    Returns the evolution of the multiplicity fraction or companion frequency for a selected primary mass, either the fraction at a time or only for stars that dont accrete more.

    Inputs
    ----------
    file: list of sinkdata objects
    The original file before system assignment.

    Master_File: list of list of star system objects
    All of the systems for the original file.

    Parameters
    ----------
    steps : int,optional
    The number of snapshots to include in one step. By default, it is 1 meaning every snapshot.

    read_in_result :bool,optional
    Whether to read in results or perform system assignment for each snapshot.

    start : bool,optional
    First snapshot to look at. By default, it is the first snapshot with stars of the target mass.
        
    target_mass: int,float,optional
    The target primary mass to consider.

    upper_limit: int,float,optional
    The highest allowed mass for the primary.

    lower_limit: int,float,optional
    The lowest allowed mass for the primary.

    rolling_avg: bool,optional
    Whether to employ a rolling average.

    rolling_window_Myr: int,float,optional
    How much time to include in the rolling window. [in Myr]
    
    time_norm : string,optional
    Which normalization to use for time (Myr,tff,atff (alpha tff))

    multiplicity: string,optional
    Which multiplicity property to use (MF,CF)

    Returns
    -------
    time: list
    The times in the simulation (in free fall time).

    fraction: list
    The multiplicity fraction of target mass primaries at any time.

    consistent_fraction: list
    The multiplicity fraction of target mass primaries that stay the same mass at any time.

    Example
    -------
    Multiplicity_Fraction_Time_Evolution(M2e4_C_M_2e7,M2e4_C_M_2e7_systems,'M2e4_C_M_2e7') 
    Plots the multiplicity time fraction over the runtime of the simulation.
    '''
    consistent_solar_mass = []
    consistent_solar_mass_unb = []
    rolling_window = int((int(time_to_snaps(rolling_window_Myr,file))//2)*2+1)
    if read_in_result == False:
        last_snap = system_creation(file,-1) #Getting the primaries in the last snap
        steps = steps
    elif read_in_result == True:
        last_snap = Master_File[-1]
        steps = 1
    #Getting a list of primaries that stay around the target mass at the end
    for i in last_snap:
        if lower_limit<=i.primary<=upper_limit:
            if i.no>1:
                consistent_solar_mass.append(i.primary_id)
            else:
                consistent_solar_mass_unb.append(i.primary_id)
    fraction = []; fraction_err = [] #This fraction comes without ignoring the primaries that change mass
    fraction1 = []; fraction1_err = []  #This fraction checks that the primaries are at a consistent mass
    time = []
    start = Mass_Creation_Finder(file,min_mass = lower_limit)
    #this one gets the masses and finishes off the graph of the consistent primaries
    for i in tqdm(range(start,len(file),steps),desc = 'By Snapshot',position=0):
        if read_in_result == False:
            sys = system_creation(file,i)
        elif read_in_result == True:
            sys = Master_File[i]
        primary_count = 0
        other_count = 0
        primary_easy = 0
        full_count = 0
        companion_count = 0
        companion_easy = 0
        for j in sys:
            if lower_limit<=j.primary<=upper_limit:
                if j.no>1 and j.primary_id in consistent_solar_mass:
                    primary_count = primary_count + 1
                    companion_count = companion_count + j.no - 1
                elif j.no == 1 and j.primary_id in consistent_solar_mass_unb:
                    other_count = other_count + 1
                full_count+=1
                if j.no >1:
                    primary_easy+=1
                    companion_easy+= j.no-1
        if multiplicity == 'MF':
            if primary_count == 0 and other_count == 0:
                fraction1.append(np.nan)
                fraction1_err.append(np.nan)
            else:
                fraction1.append(primary_count/(primary_count+other_count))
                fraction1_err.append(Psigma(primary_count+other_count,primary_count))
            if primary_easy == 0 and full_count == 0:
                fraction.append(np.nan)
                fraction_err.append(np.nan)
            else:
                fraction.append(primary_easy/full_count)
                fraction_err.append(Psigma(full_count,primary_easy))
        elif multiplicity == 'CF':
            if companion_count == 0 and other_count == 0:
                fraction1.append(np.nan)
                fraction1_err.append(np.nan)
            else:
                fraction1.append(companion_count/(companion_count+other_count))
                fraction1_err.append(Lsigma(companion_count+other_count,companion_count))
            if companion_easy == 0 and full_count == 0:
                fraction.append(np.nan)
                fraction_err.append(np.nan)
            else:
                fraction.append(companion_easy/full_count)
                fraction_err.append(Lsigma(full_count,companion_easy))
        time.append(file[i].t)
    time = np.array(time)
    ff_t = t_ff(file_properties(filename,param = 'm'),file_properties(filename,param = 'r'))
    if time_norm == 'atff':
        time = (time/(ff_t*np.sqrt(file_properties(filename,param = 'alpha'))))
    elif time_norm == 'tff':
        time = time/ff_t
    else:
        time = time*code_time_to_Myr
    if plot == True:
        plt.figure(figsize=(6,6))
        if time_norm == 'atff':
            plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
        elif time_norm == 'tff':
            plt.xlabel(r'Time [$t_\mathrm{ff}$]')
        elif time_norm == 'Myr':
            plt.xlabel('Time [Myr]')
        if multiplicity == 'MF':
            plt.ylabel('Multiplicity Fraction')
            plt.ylim([-0.05,1.05])
        elif multiplicity == 'CF':
            plt.ylabel('Companion Frequency')
            plt.ylim([-0.1,3.1])
        if rolling_avg is True:
            #time = rolling_average(time,rolling_window)
            fraction = rolling_average(fraction,rolling_window)
            fraction_err = rolling_average(fraction+fraction_err,rolling_window) -rolling_average(fraction,rolling_window)
            fraction1 = rolling_average(fraction1,rolling_window)
            fraction1_err = rolling_average(fraction1+fraction1_err,rolling_window)- rolling_average(fraction1,rolling_window)
        #plt.plot(time,fraction,label = 'All stars', linestyle='--')
        #plt.fill_between(time, fraction-fraction_err, y2=fraction+fraction_err,alpha=0.25)
        plt.plot(time,fraction1,label = 'Stars no longer accreting', linestyle='-')
        plt.fill_between(time, fraction1-fraction1_err, y2=fraction1+fraction1_err,alpha=0.25)
        if target_mass == 1:
            if multiplicity == 'MF':
                plt.errorbar(max(time)*0.9,0.46,yerr=0.03,marker = 'o',capsize = 5,color = 'black',label = 'Observations',ls='none')
            elif multiplicity == 'CF':
                plt.errorbar(max(time)*0.9,0.6,yerr=0.02,marker = 'o',capsize = 5,color = 'black',label = 'Observations',ls='none')
        elif target_mass == 10:
            if multiplicity == 'MF':
                plt.errorbar(max(time)*0.9,0.93,lolims = True,yerr = 0.04,marker = 'o',capsize = 5,color = 'black',label = 'Observations',ls='none')
            elif multiplicity == 'CF':
                plt.errorbar(max(time)*0.9,1.8,lolims = True,yerr = 0.4,marker = 'o',capsize = 5,color = 'black',label = 'Observations',ls='none')
        plt.legend(loc=3,fontsize = 14)
        plt.text(0.99,0.9,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ r' $\mathrm{M_\odot}$',transform = plt.gca().transAxes,horizontalalignment = 'right',fontsize=14)
        #plt.text(0.7,0.4,str(filename),transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
    elif plot == False:
        return time,np.array(fraction),np.array(fraction_err), np.array(fraction1),np.array(fraction1_err)

def YSO_multiplicity(file,Master_File,min_age = 0,target_age = 0.5,start = 1000, return_all=False):
    '''
    The multiplicity fraction of all objects in a certain age range.

    Inputs
    ----------
    file: list of sinkdata objects
    The original file before system assignment.

    Master_File: list of list of star system objects
    All of the systems for the original file.

    Parameters
    ----------
    min_age : int,float,optional
    The minimum age of objects.

    target_age :int,float,optional
    The maximum age of the objects

    Returns
    -------
    multiplicity: list
    The multiplicity fraction of the objects in the age range.

    object_count: list
    The number of objects in the age range.

    average_mass: list
    The average mass of the objects in the age range.

    Example
    -------
    YSO_multiplicity(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems)
    '''    
    form = []
    for i in range(0,len(file)):
        if 'ProtoStellarAge' in file[i].extra_data.keys():
            form.append(-1)
        else:
            form.append(1)
    multiplicity = []
    multiplicity_err = []
    bin_count = []
    bound_count=[]
    average_mass = []
    for k in tqdm(range(len(Master_File)),position = 0):
        current_time = file[k].t*code_time_to_Myr
        pcount = 0
        ubcount = 0
        tot_mass = 0
        for j in Master_File[k]:
            age_checker = 0
            age = 0
            for Id in j.ids:
                if form[k] == -1:
                    age = (current_time-j.formation_time_Myr[j.ids == Id])[0]
                    if min_age<=age<=target_age:
                        age_checker += 1
                elif form[k] == 1:
                    first_snap = first_snap_finder(Id,file)
                    form_time = file[first_snap].formation_time[file[first_snap].id == Id]
                    age = (current_time - form_time)*code_time_to_Myr
                    if min_age<=age<=target_age:
                        age_checker += 1
            semaxis = smaxis(j)/m_to_AU
            if age_checker == j.no and j.no>1 and 20.0<=semaxis<=10000.0:
                pcount += 1
                tot_mass += j.primary 
            elif age_checker == j.no and j.no == 1:
                ubcount+= 1
                tot_mass += j.primary 
        if pcount+ubcount == 0:
            multiplicity.append(np.nan)
            multiplicity_err.append(np.nan)
            average_mass.append(np.nan)
        else:
            multiplicity.append(pcount/(pcount+ubcount))
            multiplicity_err.append(Psigma(pcount+ubcount,pcount))
            average_mass.append(tot_mass/(pcount+ubcount))
        bin_count.append(pcount+ubcount) 
        bound_count.append(pcount)
    if return_all:
        return np.array(multiplicity),np.array(multiplicity_err),np.array(bin_count),np.array(average_mass), np.array(bound_count)
    else:
        return np.array(multiplicity),np.array(multiplicity_err),np.array(bin_count),np.array(average_mass)

#This function tracks the evolution of different stars over their lifetime
def star_multiplicity_tracker(file,Master_File,T = 2,dt = 0.5,read_in_result = True,plot = False,target_mass = 1,upper_limit = 1.3,lower_limit = 0.7,zero = 'Formation',steps = 1,select_by_time = True,random_override = False,manual_random = False,sample_size = 20):
    '''
    The status of stars born in a certain time range tracked throughout their lifetime in the simulation.

    Inputs
    ----------
    file: list of sinkdata objects
    The original file before system assignment.

    Master_File: list of list of star system objects
    All of the systems for the original file.

    Parameters
    ----------
    T : int,float,optional
    The time that the stars are born at.

    dt :int,float,optional
    The tolerance of the birth time. For example, if the simulation runs for 10 Myrs, T = 2 and dt = 0.5, it will choose stars born between 7.75 and 8.25 Myrs.

    read_in_result: bool,optional
    Whether to perform system assignment or use the already assigned system.

    plot: bool,optional
    Whether to return the times and multiplicities or plot them.

    target_mass: int,float,optional
    The target mass of primary to look at

    upper_limit: int,float,optional
    The upper limit of the target mass range

    lower_limit: int,float,optional
    The lower limit of the target mass range

    steps: int,optional
    The number of snapshots in one bin. If reading by result, this defaults to looking at every snapshot.

    select_by_time: bool,optional:
    Whether to track all stars or only those in a time frame.

    random_override: bool,optional:
    If you want to control a random sampling. By default, it does look at a random sample of over the sample size.

    manual_random: bool,optional
    Your choice to look at a random sample or not (only for plotting).

    sample_size: int,optional
    The amount of random stars to track (only to plot).

    zero: string,optional
    Whether to take the zero point as when the star was formed or stopped accreting. Use 'Formation' or 'Consistent Mass'.
    
    Returns
    -------
    all_times: list of lists
    The times for each of the stars all in one list.

    all_status: list of lists
    The status of each of the stars at each time. If the status is -1, it is a companion, 0, it is single, otherwise it is a primary with status denoting the number of companions

    ids: list
    The id of each of the stars.

    maturity_times: list
    The time that each star stops accreting.

    Tend:list
    The end time for each star.

    birth_times:list
    The formation times for each star.

    kept:int
    The number of stars in the selected range
    
    average_dens: int,float
    The average density in the selected range
    
    average_mass_dens: int,float
    The average mass density in the selected range

    Example
    -------
    star_multiplicity_tracker(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems,T = 2,dt = 0.33)
    '''  
    
    consistent_solar_mass = []
    if read_in_result == False:
        last_snap = system_creation(file,-1) #Getting the primaries in the last snap
        steps = steps
    elif read_in_result == True:
        last_snap = Master_File[-1]
        steps = 1
    birth_times = []; densities = []; mass_densities = []; mass_accretion_times = []
    #Getting a list of primaries that stay around the target mass at the end
    for s in last_snap:
        if lower_limit<=s.primary<=upper_limit:
            consistent_solar_mass.append(s.primary_id)  
            birth_times.append(s.formation_time_Myr[0])
            densities.append(s.init_star_vol_density[0])
            mass_densities.append(s.init_star_mass_density[0])
            mass_acr_snap = first_snap_mass_finder(s.primary_id,file,lower_limit,1e3)
            if not(mass_acr_snap is None):
                mass_accretion_times.append(file[first_snap_mass_finder(s.primary_id,file,lower_limit,upper_limit)].t*code_time_to_Myr)
            else:
                mass_accretion_times.append(file[-1].t*code_time_to_Myr) #Not sure how this could ever be executed
    birth_times = np.squeeze(np.array(birth_times)); mass_densities = np.array(mass_densities); densities = np.array(densities); mass_accretion_times = np.array(mass_accretion_times)
    selected_ind = np.full(len(birth_times),True)
    if select_by_time == True:
        Tend = file[-1].t*code_time_to_Myr
        print('Filtering by formation time')
        selected_ind = selected_ind & (birth_times<=(T+dt/2)) &  (birth_times>=(T-dt/2))
    selected_ind = np.arange(len(selected_ind))[selected_ind]
    kept = len(selected_ind)
    average_dens = np.median(densities[selected_ind])
    average_mass_dens = np.median(mass_densities[selected_ind])
    #Adjusting zero time point
    if zero == 'Consistent Mass':
        zero_times = mass_accretion_times
    elif zero == 'Formation':
        zero_times = birth_times
    snaptimes = np.array([d.t*code_time_to_Myr for d in file])
    all_times = []
    for ind in selected_ind:
        all_times.append(snaptimes[snaptimes>=zero_times[ind]]-zero_times[ind])
    time_short = [(time_array[1:]+time_array[:-1])/2 for time_array in all_times]   
    
        
    #This is quite inefficient, needs rewriting
    all_status = []
    change_in_status = []
    for ind in selected_ind:
        ID = consistent_solar_mass[ind]
        statuses = []
        start_snap = np.argmax(snaptimes>=zero_times[ind])
        for k in range(start_snap,len(file),1):
            status = 0
            if read_in_result == False:
                sys = system_creation(file,k)
            elif read_in_result == True:
                sys = Master_File[k]
            for l in sys:
                if ID in l.ids:
                    if len(l.ids)>1:
                        if ID == l.primary_id:
                            status = len(l.ids)-1
                        else:
                            status = -1
                    else:
                        status = 0
            statuses.append(status)
        all_status.append(statuses)
        change_in_status.append(np.diff(statuses))

    ids = np.array(consistent_solar_mass,dtype=np.int64)[selected_ind]
    if plot == True:
        if len(all_status)>sample_size:
            rand = True
        else:
            rand = False
        if random_override == True:
            rand = manual_random
        if rand == False:
            plt.figure(figsize=(6,6))
            plt.xlim(-0.01,max(flatten(all_times))*1.1)
            offset = np.linspace(-0.3,0.3,len(all_status))
            for i in range(len(all_status)):
                plt.plot(all_times[i],all_status[i]+offset[i],label = ids[i])
            plt.legend(loc = 'best',bbox_to_anchor=(1.1, 1, 0, 0),fontsize = 14)
            #plt.text(max(flatten(all_times))*0.8,-0.5,Files_key[n],fontsize = 12)
            plt.text(max(flatten(all_times))*0.8,-0.75,'Target Mass = '+str(target_mass),fontsize = 12)
            plt.xlabel('Time (in Myrs)')
            plt.ylabel('Status')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            #plt.savefig(Files_key[n]+'_'+str(target_mass)+'.png')
            plt.show()
            plt.figure(figsize=(6,6))
            plt.xlim(-0.01,max(flatten(time_short))*1.1)
            offset = np.linspace(-0.3,0.3,len(change_in_status))
            for i in range(len(change_in_status)):
                plt.plot(time_short[i],change_in_status[i]+offset[i],label = ids[i])
            plt.legend(loc = 'best',bbox_to_anchor=(1.1, 1, 0, 0),fontsize = 14)
            #plt.text(max(flatten(time_short))*0.8,-0.5,Files_key[n],fontsize = 12)
            plt.text(max(flatten(time_short))*0.8,-0.75,'Target Mass = '+str(target_mass),fontsize = 12)
            plt.xlabel('Time (in Myrs)')
            plt.ylabel('Change in status')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            #plt.savefig(Files_key[n]+'_'+str(target_mass)+'_Change.png')
            plt.show()
        elif rand == True:
            random_indices = random.sample(range(len(change_in_status)),sample_size)
            rand_times = []
            rand_status = []
            rand_ids = []
            for i in random_indices:
                rand_times.append(all_times[i])
                rand_status.append(all_status[i])
                rand_ids.append(ids[i])
            plt.figure(figsize = (20,20))
            plt.xlim(-0.01,max(flatten(rand_times))*1.1)
            offset = np.linspace(-0.3,0.3,len(rand_status))
            for i in range(len(rand_times)):
                plt.plot(rand_times[i],rand_status[i]+offset[i],label = rand_ids[i])
            plt.legend(loc = 'best',bbox_to_anchor=(1.1, 1, 0, 0),fontsize = 14)
            plt.xlabel('Time (in Myrs)')
            plt.ylabel('Status')
            #plt.text(max(flatten(rand_times))*0.8,-0.5,Files_key[n],fontsize = 12)
            plt.text(max(flatten(rand_times))*0.8,-0.75,'Target Mass = '+str(target_mass),fontsize = 12)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            #plt.savefig(Files_key[n]+'_'+str(target_mass)+'.png')
            plt.show()
            random_indices = random.sample(range(len(change_in_status)),sample_size)
            rand_times = []
            rand_status = []
            rand_ids = []
            for i in random_indices:
                rand_times.append(time_short[i])
                rand_status.append(change_in_status[i])
                rand_ids.append(ids[i])
            plt.figure(figsize = (20,20))
            plt.xlim(-0.01,max(flatten(rand_times))*1.1)
            offset = np.linspace(-0.3,0.3,len(rand_status))
            for i in range(len(rand_times)):
                plt.plot(rand_times[i],rand_status[i]+offset[i],label = rand_ids[i])
            plt.legend(loc = 'best',bbox_to_anchor=(1.1, 1, 0, 0),fontsize = 14)
            plt.xlabel('Time (in Myrs)')
            plt.ylabel('Change in Status')
            #plt.text(max(flatten(rand_times))*0.8,-0.5,Files_key[n],fontsize = 12)
            plt.text(max(flatten(rand_times))*0.8,-0.75,'Target Mass = '+str(target_mass),fontsize = 12)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            #plt.savefig(Files_key[n]+'_'+str(target_mass)+'_Change.png')
            plt.show()

    if plot == False:
        placeholder = 0
        placeholder2 = 0
        placeholder3 = 0
        placeholder4 = 0
        if select_by_time == True:
            placeholder = Tend
            placeholder2 = kept
            placeholder3 = average_dens
            placeholder4 = average_mass_dens
        return all_times,all_status,ids,zero_times,placeholder,birth_times,placeholder2,placeholder3,placeholder4

#This function gives the multiplicity fraction at different ages
def MFCF_and_age(file,Master_File,T = 2,dt = 0.5,target_mass = 1,upper_limit = 1.5,lower_limit = 1/1.5,read_in_result = True,select_by_time = True,zero = 'Formation',plot = True,steps = 1):
    '''
    The average multiplicity fraction of stars born in a certain time range tracked throughout their lifetime in the simulation.

    Inputs
    ----------
    file: list of sinkdata objects
    The original file before system assignment.

    Master_File: list of list of star system objects
    All of the systems for the original file.

    Parameters
    ----------
    T : int,float,optional
    The time that the stars are born at.

    dt :int,float,optional
    The tolerance of the birth time. For example, if the simulation runs for 10 Myrs, T = 2 and dt = 0.5, it will choose stars born between 7.75 and 8.25 Myrs.

    target_mass: int,float,optional
    The target mass of primary to look at

    upper_limit: int,float,optional
    The upper limit of the target mass range

    lower_limit: int,float,optional
    The lower limit of the target mass range

    read_in_result: bool,optional
    Whether to perform system assignment or use the already assigned system.

    select_by_time: bool,optional:
    Whether to track all stars or only those in a time frame.

    zero: string,optional
    Whether to take the zero point as when the star was formed or stopped accreting. Use 'Formation' or 'Consistent Mass'.

    plot: bool,optional
    Whether to return the times and multiplicities or plot them.

    steps: int,optional
    The number of snapshots in one bin. If reading by result, this defaults to looking at every snapshot.

    Returns
    -------
    age_bins: array
    The age over which the stars are in.

    multiplicity: array
    The average multiplicity fraction of the objects in the bins.

    birth_times:list
    The birth times of the stars.

    kept:int
    The number of stars in the age range.

    average_dens: int,float
    The average density in the selected range
    
    average_mass_dens: int,float
    The average mass density in the selected range

    Example
    -------
    multiplicity_frac_and_age(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems)
    '''  
    times,status,ids,maturity_times,Tend,birth_times,kept,average_dens,average_mass_dens = star_multiplicity_tracker(file,Master_File,T = T,dt = dt,read_in_result = read_in_result,plot = False,target_mass = target_mass,upper_limit=upper_limit,lower_limit=lower_limit,zero = zero,steps = steps,select_by_time=select_by_time)
    counted_all = []
    is_primary_all = []
    time_all = []; status_all = []
    lengths = []
    for t,s in zip(times,status):
        time_all+=list(t);status_all+=list(s)
        lengths.append(len(t))
    time_all = np.array(time_all);status_all = np.array(status_all)
    age_bins=np.linspace(0,max(time_all),max(lengths)+1)
    counted_all = status_all.copy();counted_all[status_all>=0] = 1;counted_all[status_all<0] = 0
    is_primary_all = status_all.copy();is_primary_all[status_all>0] = 1;is_primary_all[status_all<=0] = 0
    comp_all = status_all.copy();comp_all[status_all<=0] = 0
    counted_in_bin, temp = np.histogram(time_all, bins=age_bins, weights=counted_all)
    is_prmary_in_bin, temp = np.histogram(time_all, bins=age_bins, weights=is_primary_all)
    no_comp_in_bin,temp = np.histogram(time_all, bins=age_bins, weights=comp_all)
    MF_in_bin = (is_prmary_in_bin)/(counted_in_bin)
    CF_in_bin = (no_comp_in_bin)/(counted_in_bin)
    
    age_bins_mean = (age_bins[1:] + age_bins[:-1])/2
    times = []
    for i in file:
        times.append(i.t*code_time_to_Myr)
    if plot == True:
        plt.figure(figsize=(6,6))
        if select_by_time == True:
            new_stars_count(file)
            plt.fill_between([T-dt/2,T+dt/2],16,alpha = 0.3)
            plt.xlabel('Simulation Time [Myr]')
            plt.ylabel('No of New Stars')
            plt.show()
            plt.figure(figsize=(6,6))
            new_stars_count(file,lower_limit=lower_limit,upper_limit=upper_limit)
            plt.fill_between([T-dt/2,T+dt/2],8,-2,alpha = 0.3)
            plt.xlabel('Simulation Time [Myr]')
            plt.ylabel('Change in # of Target Mass Stars')
            plt.show()
            plt.figure(figsize=(6,6))
            plt.plot(age_bins_mean[age_bins_mean<(times[-1]-(T+dt/2))],MF_in_bin[age_bins_mean<(times[-1]-(T+dt/2))])
            plt.ylim([-0.1,1.1])
            plt.xlabel('Age in Myrs')
            plt.ylabel('Average Multiplicity Fraction')
            plt.text(0.1,0.7,'Target Mass ='+str(target_mass),transform = plt.gca().transAxes)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            plt.show()
            plt.figure(figsize=(6,6))
            plt.plot(age_bins_mean[age_bins_mean<(times[-1]-(T+dt/2))],CF_in_bin[age_bins_mean<(times[-1]-(T+dt/2))])
            plt.ylim([-0.1,3.1])
            plt.xlabel('Age in Myrs')
            plt.ylabel('Average Companion Frequency')
            plt.text(0.1,0.7,'Target Mass ='+str(target_mass),transform = plt.gca().transAxes)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            plt.show()
        else:
            plt.plot(age_bins_mean,MF_in_bin)
        #plt.plot(age_bins_mean,multiplicity_in_bin,label = 'Multiplicity at Age Plot')
        #plt.legend(fontsize = 14)
        plt.figure(figsize=(6,6))
        if select_by_time == True:
            plt.plot(age_bins_mean[age_bins_mean<(times[-1]-(T+dt/2))],(counted_in_bin)[age_bins_mean<(times[-1]-(T+dt/2))])
        else:
            plt.plot(age_bins_mean,(counted_in_bin))
        plt.xlabel('Age in Myrs')
        plt.ylabel('Number of Stars')
        #plt.legend(fontsize = 14)
        plt.show()
    else:
        return age_bins_mean,[MF_in_bin,CF_in_bin],birth_times,kept,average_dens,average_mass_dens,is_prmary_in_bin[age_bins_mean<(times[-1]-(T+dt/2))][-1],counted_in_bin[age_bins_mean<(times[-1]-(T+dt/2))][-1]

def Orbital_Plot_2D(system,plot = True):
    '''Create an orbital plane projection plot of any system'''
    if system.no>1:
        #Getting the velocity and coordinates of the CoM (of only secondary and primary)
        com_coord = (system.primary*system.x[np.array(system.ids) == system.primary_id]+system.secondary*system.x[system.m == system.secondary])/(system.primary+system.secondary)
        com_vel = (system.primary*system.v[np.array(system.ids) == system.primary_id]+system.secondary*system.v[system.m == system.secondary])/(system.primary+system.secondary)
        #Getting the mass, coordiantes and velocity in the CoM frame
        m1 = system.primary
        m2 = system.secondary
        r1 = system.x[np.array(system.ids) == system.primary_id] - com_coord
        r2 = system.x[np.array(system.m) == system.secondary] - com_coord
        v1 = system.v[np.array(system.ids) == system.primary_id] - com_vel
        v2 = system.v[np.array(system.m) == system.secondary] - com_vel
        #Calculating the angular momentum and normalizing it
        L = m1*(np.cross(r1,v1)) + m2*(np.cross(r2,v2)) #Check with x and y
        L = L[0]
        l = np.linalg.norm(L)
        L_unit = L/l
        #Finding the two unit vectors
        unit_vector = [1,0,0]
        if L_unit[0]>0 and L_unit[1] == 0 and L_unit[2] == 0:
            unit_vector = [0,1,0]
        e1_nonnorm = np.cross(L_unit,unit_vector) #Check the 0th component
        e1 = e1_nonnorm/np.linalg.norm(e1_nonnorm) #Check if this is proper (Dont use ex or ey)
        e2 = np.cross(e1,L_unit) #Check that it is a proper shape
        #Getting the CoM of the whole system coordinates
        com_cord_all_nonnorm = 0
        com_vel_all_nonnorm = 0
        for i in range(system.no):
            com_cord_all_nonnorm += system.m[i]*system.x[i]
            com_vel_all_nonnorm += system.m[i]*system.v[i]
        com_cord_all = com_cord_all_nonnorm/sum(system.m)
        com_vel_all = com_vel_all_nonnorm/sum(system.m)
        #Now getting the x and v in the CoM frame
        com_frame_x = system.x - com_cord_all
        com_frame_v = system.v - com_vel_all
        #Finally, we project the x and v to the orbital plane
        x_new = np.zeros((system.no,2))
        v_new = np.zeros((system.no,2))
        for i in range(system.no):
            x_new[i][0] = np.dot(com_frame_x[i],e1)
            x_new[i][1] = np.dot(com_frame_x[i],e2)
            v_new[i][0] = np.dot(com_frame_v[i],e1)
            v_new[i][1] = np.dot(com_frame_v[i],e2)
        #Now we plot onto a quiver plot 
        if plot == True:
            plt.figure(figsize = (6,6))
            for i in range(system.no):
                plt.quiver(x_new[i][0]*pc_to_AU,x_new[i][1]*pc_to_AU,v_new[i][0],v_new[i][1])
                plt.scatter(x_new[i][0]*pc_to_AU,x_new[i][1]*pc_to_AU,s = system.m[i]*100)
            plt.xlabel('Coordinate 1 [AU]')
            plt.ylabel('Coordinate 2 [AU]')
            plt.show()
        #Checking original KE
        oKE = 0
        for i in range(system.no):
            oKE += 0.5*system.m[i]*np.linalg.norm(com_frame_v[i])**2
        #Checking new KE
        com_new_vel_all_nonnorm = 0
        for i in range(system.no):
            com_new_vel_all_nonnorm += system.m[i]*v_new[i]
        com_new_vel = np.array(com_new_vel_all_nonnorm)/sum(system.m)
        v_new_com_frame = np.zeros((system.no,2))
        for i in range(system.no):
            v_new_com_frame[i] = v_new[i] - com_new_vel
        nKE = 0
        for i in range(system.no):
            nKE += 0.5*system.m[i]*np.linalg.norm(v_new_com_frame[i])**2
        if plot == True:
            print("1 - KE'/KE = "+str((1-nKE/oKE).round(2)))
            print('Semi Major Axis of system is ' +str((smaxis(system)/m_to_AU).round(2))+ ' AU')
        if plot == False:
            return 1-nKE/oKE
    elif system.no == 1:
        if plot == True:
            print('No, this is a plot of one image')
        else:
            return np.nan

def smaxis_tracker(file,Master_File,system_ids,plot = True,KE_tracker = False):
    '''
    Tracking the semi-major axis between some ids throughout the simulation runtime.
    Inputs
    ----------
    file: list of sinkdata objects
    The original file before system assignment.

    Master_File: list of list of star system objects
    All of the systems for the original file.

    system_ids: list
    The ids to track the semi-major axis of.

    Parameters
    ----------
    plot: bool,optional
    Whether to return the values or plot them.

    KE_tracker: bool,optional
    Whether to also look at the loss in kinetic energy from the orbital plane projection.

    Returns
    -------
    smaxes: list
    The semi major axis of the given ids throughout the simulation

    no_of_stars: list
    The number of stars in the system throughout the simulation

    KE_tracks: list
    The loss in KE from an orbital projection throughout time. Only returns this if KE_tracker is true.

    times: list
    The times that the system exists in the simulation.

    Example
    -------
    smaxis_tracker(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems,[112324.0,1233431.0])
    '''  

    smaxes = []
    times = []
    no_of_stars = []
    f_tracks = []
    for i in range(len(Master_File)):
        marker = 0
        times.append(file[i].t*code_time_to_Myr)
        for j in Master_File[i]:
            if(set(system_ids).issubset(set(j.ids))):
                smaxes.append(np.log10(smaxis(j)/m_to_AU))
                no_of_stars.append(j.no)
                if KE_tracker == True:
                    f_tracks.append(Orbital_Plot_2D(j,plot = False))
                marker = 1
        if marker == 0:
            smaxes.append(np.nan)
            no_of_stars.append(np.nan)
            f_tracks.append(np.nan)

    if plot == True:
        plt.figure(figsize = (6,6))
        plt.plot(times,smaxes)
        plt.xlabel('Time (Myr)')
        plt.ylabel('Log Semi Major Axis (AU)')
        plt.show()

        plt.figure(figsize = (6,6))
        plt.plot(times,no_of_stars)
        plt.xlabel('Time (Myr)')
        plt.ylabel('No of Stars in System')
        plt.show()
        
        if KE_tracker == True:
            plt.figure(figsize = (6,6))
            plt.plot(times,f_tracks)
            plt.xlabel('Time (Myr)')
            plt.ylabel('Kinetic Energy Loss in Orbital Projection')
            plt.show()
            
        
    elif plot == False:
        if KE_tracker == False:
            return smaxes,no_of_stars,times
        elif KE_tracker == True:
            return smaxes,no_of_stars,f_tracks,times

def distance_tracker_binaries(file,Master_File,system_ids,plot = True,rolling_avg = True,rolling_window_Myr = 0.1):
    '''
    Tracking the distance axis between some ids in a binary system throughout the simulation runtime.
    Inputs
    ----------
    file: list of sinkdata objects
    The original file before system assignment.

    Master_File: list of list of star system objects
    All of the systems for the original file.

    system_ids: list
    The ids to track the semi-major axis of.

    Parameters
    ----------
    plot: bool,optional
    Whether to return the values or plot them.

    Returns
    -------
    smaxes: list
    The distance between the given ids throughout the simulation

    times: list
    The times that the system exists in the simulation (0 is when the system formed).

    Example
    -------
    distance_tracker_binaries(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems,[112324.0,1233431.0])
    '''  

    if len(system_ids) != 2:
        print('Please provide a binary system')
        return
    distances = []
    times = []
    snaps = []
    for i in range(len(Master_File)):
        marker = 0
        times.append(file[i].t*code_time_to_Myr)
        pos1 = file[i].x[file[i].id == system_ids[0]]
        pos2 = file[i].x[file[i].id == system_ids[1]]
        distances.append(np.log10(np.linalg.norm(pos1-pos2)*pc_to_AU))
        for j in Master_File[i]:
            if(set(system_ids).issubset(set(j.ids))):
                marker = 1
        if marker == 1:
            snaps.append(i)
    times = np.array(times)
    if len(snaps) > 0:
        t0 = file[snaps[0]].t*code_time_to_Myr
    else:
        t0 = file[-1].t*code_time_to_Myr
    times = times-t0
    
    if rolling_avg is True:
        rolling_window_Myr = time_to_snaps(rolling_window_Myr,file)
        if rolling_window_Myr%2 == 0:
            rolling_window_Myr -= 1
        rolling_window_Myr = int(rolling_window_Myr)
        #times = rolling_average(times,rolling_window_Myr)
        distances = rolling_average(distances,rolling_window_Myr)

    if plot == True:
        plt.figure(figsize = (6,6))
        plt.plot(times,distances)
        plt.xlabel('Time (Myr)')
        plt.ylabel('Log Semi Major Axis (AU)')
        plt.show()
        
    elif plot == False:
        return distances,times

def formation_distance(id_list,file_name,log = True, file=None):
    '''The formation distance between two ids with the original file name provided as a string.'''
    if (file_name is None):
        Brown_Dwarf_File = file
    else:
        pickle_file = open(file_name +'.pickle','rb')
        Brown_Dwarf_File = pickle.load(pickle_file)
        pickle_file.close()
    
    logdist = []
    first_snap1 = first_snap_finder(id_list[0],Brown_Dwarf_File)
    first_snap2 = first_snap_finder(id_list[1],Brown_Dwarf_File)
    first_snap_both = max([first_snap1,first_snap2])
    pos1 = Brown_Dwarf_File[first_snap_both].x[Brown_Dwarf_File[first_snap_both].id == id_list[0]]
    pos2 = Brown_Dwarf_File[first_snap_both].x[Brown_Dwarf_File[first_snap_both].id == id_list[1]]

    distance = np.linalg.norm(pos1-pos2)*pc_to_AU
    if log == True:
        return np.log10(distance)
    else:
        return distance

def q_with_formation(Master_File,file_name,snapshot,limit = 10000,upper_mass_limit = 1.3,lower_mass_limit = 0.7):
    '''
    Seperating the mass ratios based on formation distance.
    Inputs
    ----------
    Master_File: list of list of star system objects
    All of the systems for the original file.

    file_name: str
    The name of the file to check the formation distance from.

    snapshot: int
    The snapshot to check.

    Parameters
    ----------
    limit: int,float,optional
    The formation distance limit that you want to split by.

    upper_mass_limit: int,float,optional
    The upper mass limit for the primaries

    lower_mass_limit: int,float,optional
    The lower mass limit for the primaries

    Returns
    -------
    q_list_under: list
    The mass ratios under the formation distance limit

    distance_list_under: list
    The formation distance distribution under the formation distance limit. 

    q_list_over: list
    The mass ratios over the formation distance limit

    distance_list_over: list
    The formation distance distribution over the formation distance limit. 

    all_dist: list
    The formation distance for everything.

    '''  
    q_list_under = []
    distance_list_under = []
    q_list_over = []
    distance_list_over = []
    all_dist = []
    for i in Master_File[snapshot]:
        if i.no > 1 and lower_mass_limit<=i.primary<=upper_mass_limit:
            for ids in i.ids:
                if ids != i.primary_id :
                    form_dist = formation_distance([ids,i.primary_id],file_name,log = False)
                    all_dist.append(form_dist)
                    if form_dist <= limit:
                        q_list_under.append(i.m[np.array(i.ids) == ids]/i.primary)
                        distance_list_under.append(form_dist)
                    if form_dist > limit:
                        q_list_over.append(i.m[np.array(i.ids) == ids]/i.primary)
                        distance_list_over.append(form_dist)
    return list(flatten(q_list_under)),distance_list_under,list(flatten(q_list_over)),distance_list_over,all_dist

#Using np.hist and moving the bins to the center of each bin
def hist(x,bins = 'auto',log =False,shift = False):
    '''
    Create a histogram
    Inputs
    ----------
    x: data
    The data to be binned

    Parameters
    ----------
    bins: int,list,str
    The bins to use.

    log: bool,optional
    Whether to return number of objects in bin or log number of objects.

    shift: bool,optional
    Whether to shift the bins to the center or not.

    Returns
    -------
    x_vals: list
    The bins.

    weights:list
    The weights of each bin

    Example
    -------
    hist(x)
    '''  
    if x is None:
        return None,None
    if log == True:
        weights,bins = np.histogram(np.log10(x),bins = bins)
    elif log == False:
        weights,bins = np.histogram(x,bins = bins)
    if shift == True:
        xvals = (bins[:-1] + bins[1:])/2
    else:
        xvals = bins
    return xvals,weights


def scatter_with_histograms(X,Y,xlabel,ylabel,filename,nbins=10,xlim=None,ylim=None,fig=None,\
                            msize=30, capsize=5, markers='o', colors='k',cmap='viridis',\
                            overtext=None,overtextcoord=[0.5,0.5], loghist=False,\
                            histx_color='b',histy_color='b',\
                            colorbar=False, cbar_limits=None,cbar_tick_labels=None,cbar_label='', fontsize=14,saveplot=True, vlines=[], hlines=[] ): 
    #copied from gdplot
    from matplotlib.ticker import NullFormatter
    # Start making the plot
    # definitions for the axes, sets size of subfigures
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    #Limits
    if not hasattr(xlim, "__iter__"):
        xlim=np.array([np.min(X),np.max(X)])
        ylim=np.array([np.min(Y),np.max(Y)])
    # creates the axes
    if fig is None:
        fig = plt.figure(1, figsize=(8, 8))
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)# We want plots and no labels on the overlapping part of the histograms
    nullfmt = NullFormatter()
    #Set log scale for histogram if needed
    if loghist:
        axHistx.set_yscale("log")
        axHisty.set_xscale("log")
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
#    # diagonal line
#    axScatter.plot([xlim[0],xlim[1]],[ylim[0],ylim[1]],'--',c='black',lw=1.5,alpha=0.25)
    if not hasattr(cbar_limits, "__iter__"):
        cbar_limits=np.array([np.min(colors),np.max(colors)])
    #Scatter plot
    im = axScatter.scatter(X , Y ,marker=markers, c=colors,edgecolors='black',cmap=cmap,norm=matplotlib.colors.Normalize(vmin=cbar_limits[0], vmax=cbar_limits[1], clip=True));
    #Set limits
    axScatter.set_xlim((xlim[0]-np.ptp(xlim)/40), xlim[1]+np.ptp(xlim)/40); axScatter.set_ylim((ylim[0]-np.ptp(ylim)/40, ylim[1]+np.ptp(ylim)/40))
    #Set label
    axScatter.set_xlabel(xlabel, fontsize=fontsize); axScatter.set_ylabel(ylabel, fontsize=fontsize);
    for vline in vlines:
        axScatter.axvline(x=vline,alpha = 0.3,color='k')
    for hline in hlines:
        axScatter.axhline(y=hline,alpha = 0.3,color='k')
#    #Add colorbar
    if colorbar:
        cb = plt.colorbar(im,boundaries=np.linspace(cbar_limits[0], cbar_limits[1], 100, endpoint=True)) # 'boundaries' is so the drawn colorbar ranges between the limits you want (rather than the limits of the data)
        #cb.set_clim(cbar_limits[0],cbar_limits[1]) # this only redraws the colors of the points, the range shown on the colorbar itself will not change with just this line alone (see above line)
        if hasattr(cbar_tick_labels, "__iter__"):
            nticks=len(cbar_tick_labels)
            cb.set_ticks(np.linspace(cbar_limits[0], cbar_limits[1], nticks, endpoint=True))
            cb.set_ticklabels(cbar_tick_labels)
        else:
            cb.set_ticks(np.linspace(cbar_limits[0], cbar_limits[1], 5, endpoint=True))
        if (cbar_label != '' and cbar_label != None):
            cb.set_label(cbar_label, fontsize=fontsize)
    axHistx.hist(X, bins=np.linspace(xlim[0],xlim[1],nbins),edgecolor='white',color=histx_color);
    axHisty.hist(Y, bins=np.linspace(ylim[0],ylim[1],nbins),edgecolor='white',color=histy_color, orientation='horizontal');
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    #Plot overtext
    if (overtext!=None):
        plt.text(overtextcoord[0], overtextcoord[1], overtext, horizontalalignment='left', verticalalignment='center', transform=axScatter.transAxes, fontsize=fontsize)
    #Save figure
    if saveplot:
        fig.savefig(filename+'.png',dpi=150, bbox_inches='tight')


def Primordial_separation_distribution(file,Master_File,upper_limit=1.3,lower_limit = 0.7,plot = True,cmap='viridis', apply_filters=True, outfilename='separation_dist', y_axis='final_dist', minmass_for_angle=0.08):
    '''
    Shows the distribution of distances between primary and companions at formation and when they form a system, done for systems that exist in the last snapshot

    Inputs
    ----------
    file: list of sinkdata objects
    The original file before system assignment.

    Master_File: list of list of star system objects
    All of the systems for the original file.

    Parameters
    ----------

    upper_limit: int,float,optional
    The upper limit of the target mass range

    lower_limit: int,float,optional
    The lower limit of the target mass range
    
    plot: bool,optional
    Plot or return the values
    
    '''  
    if apply_filters and ( not (file is None)) :
        Master_File = q_filter(Master_File)
        Master_File[-1]=full_simple_filter(Master_File,file,selected_snap=-1, filter_in_class = False)
    form_distances = []
    sys_form_distances = []; sys_form_angles = []
    q_vals = []
    final_sep = []; final_angles = []
    for system in Master_File[-1]:
        if (system.primary>=lower_limit) and (system.primary<=upper_limit) :
            for comp_id in system.ids:
                if comp_id!=system.primary_id:
                    q_vals.append(system.m[system.ids==comp_id][0]/system.primary) #store mass ratio
                    x1 = np.squeeze(system.x[system.ids==system.primary_id])
                    x2 = np.squeeze(system.x[system.ids==comp_id])
                    final_sep.append(np.linalg.norm(x1-x2))
                    final_angles.append( momentum_angle(system.primary_id,comp_id,file,system.snapshot_num) )
                    #Find the at formation separation 
                    form_distances.append(formation_distance([system.primary_id,comp_id ],None,log = False, file=file)) #this is already AU
                    #Find the separation when they become a system
                    dist = None; angle = None
                    for snap_systems in Master_File:
                        for snap_system in snap_systems:
                            if (comp_id in snap_system.ids) and (system.primary_id in snap_system.ids):
                                if (dist is None):
                                    x1 = np.squeeze(snap_system.x[snap_system.ids==system.primary_id])
                                    x2 = np.squeeze(snap_system.x[snap_system.ids==comp_id])
                                    dist = np.linalg.norm(x1-x2)
                                    sys_form_distances.append(dist) 
                                if (angle is None) and (np.squeeze(snap_system.m[snap_system.ids==comp_id])>minmass_for_angle):
                                    angle = momentum_angle(system.primary_id,comp_id,file,snap_system.snapshot_num)
                                    sys_form_angles.append(angle)
                        if (not (dist is None)) and (not (angle is None)):
                            break
    form_distances = np.log10(form_distances); sys_form_distances = np.log10(sys_form_distances)+np.log10(pc_to_AU);
    q_vals = np.array(q_vals); final_sep = np.log10(final_sep)+np.log10(pc_to_AU);
    #Now we have a list of distances at sink and system formation
    
    if plot == True:
        #Plotting the multiplicity over age
        plt.figure(figsize = (6,6))
        if upper_limit<=50:
            overtext = 'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ r' $\mathrm{M_\odot}$'
        else:
            overtext = None
        if y_axis == 'final_dist':
            ylabel='Log Distance [AU]'
            yvals=final_sep
        else:
            ylabel='Log Distance at system formation [AU]'
            yvals=sys_form_distances
        scatter_with_histograms(form_distances,yvals,'Log Distance at star formation [AU]',ylabel,'',nbins=10, msize=30, capsize=5, markers='x', colors=q_vals,cmap=cmap, histx_color='b',histy_color='b',\
                            colorbar=True, cbar_limits=[0.0,1.0],cbar_tick_labels=None,cbar_label='Mass ratio', xlim=[0.5,5.5], ylim=[0.5,5.5], fontsize=14,saveplot=False,overtext=overtext,overtextcoord=[0.01,0.9],vlines=[np.log10(20)],hlines=[np.log10(20)] ) 
        adjust_font(fig=plt.gcf(), ax_fontsize=14)
        plt.savefig(outfilename+'.png',dpi = 150, bbox_inches='tight')
        plt.show()
        plt.close()
        #Let's do a plot of the primordial separation angle
        plt.figure(figsize = (6,6))
        if upper_limit<=50:
            overtext = 'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ r' $\mathrm{M_\odot}$'
        else:
            overtext = None
        yvals=sys_form_distances
        scatter_with_histograms(final_angles,sys_form_angles,'Misalignment angle [°]','Primordial misalignment angle','',nbins=10, msize=30, capsize=5, markers='x', colors=q_vals,cmap=cmap, histx_color='b',histy_color='b',\
                            colorbar=True, cbar_limits=[0.0,1.0],cbar_tick_labels=None,cbar_label='Mass ratio', xlim=[0,180], ylim=[0,180], fontsize=14,saveplot=False,overtext=overtext,overtextcoord=[0.01,0.9] ) 
        adjust_font(fig=plt.gcf(), ax_fontsize=14)
        plt.savefig(outfilename+'_angle.png',dpi = 150, bbox_inches='tight')
        plt.show()
        plt.close()
        
  
    return  form_distances, sys_form_distances,final_angles, sys_form_angles, q_vals


def multiplicity_vs_formation(file,Master_File,edges = None,upper_limit=1.3,lower_limit = 0.7,target_mass = None,zero = 'Formation',multiplicity = 'MF',filename = None,min_time_bin = 0.2,adaptive_no = 20,x_axis = 'time',plot = True,label=None, color = 'k'):
    '''
    The average multiplicity of stars born in certain time ranges tracked throughout their lifetime in the simulation.

    Inputs
    ----------
    file: list of sinkdata objects
    The original file before system assignment.

    Master_File: list of list of star system objects
    All of the systems for the original file.

    Parameters
    ----------
    T_list : list,optional
    The time that the stars are born at.

    dt_list :list,optional
    The tolerance of the birth time.

    upper_limit: int,float,optional
    The upper limit of the target mass range

    lower_limit: int,float,optional
    The lower limit of the target mass range
    
    target_mass: int,float,optional
    The target mass of primary to look at
    
    zero: string,optional
    Whether to take the zero point as when the star was formed or stopped accreting. Use 'Formation' or 'Consistent Mass'.

    multiplicity: bool,optional
    Whether to plot for the multiplicity properties, multiplicity fraction or Companion Frequency.
    
    filename: str,optional
    The name of the file. Will be printed on plot if provided.
    
    min_time_bin: int,optional
    The minimum time bin to plot on the time histogram
    
    adaptive_binning: bool,optional
    Whether to adopt adaptive binning (same no of stars in each bin)
    
    adaptive_no: int,optional
    The number of stars in each bin
    
    x_axis: string,optional
    Whether to plot the MF/CF with the formation time/density/mass density
    
    plot: bool,optional
    Plot or return the values
    
    Returns
    -------
    age_bins: array
    The age over which the stars are in.

    multiplicity: array
    The average multiplicity fraction of the objects in the bins.

    Example
    -------
    multiplicity_vs_formation(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems,adaptive_binning = True,adaptive_no = 20)
    '''  
  
    if x_axis == 'time':
        xvals = np.array([s.formation_time_Myr[np.argmax(s.m)] for s in Master_File[-1]])
        x_label = 'Formation time[Myr]'
    elif x_axis == 'density' or x_axis == 'mass density': 
        if x_axis == 'density':
            density = 'number'
            x_label = r'Log Stellar density at formation [$\mathrm{pc}^{-3}$]'
        elif x_axis == 'mass density':
            density = 'mass'
            x_label = r'Log Stellar mass density at formation [$\mathrm{M_\odot}\mathrm{pc}^{-3}$]'
        xvals = np.log10([s.init_density[density][np.argmax(s.m)] for s in Master_File[-1]])  
    primary_mass = np.array([s.primary for s in Master_File[-1]])
    index = (primary_mass>=lower_limit) & (primary_mass<=upper_limit) #the ones with the right primary masses
    xvals = xvals[index];
    companion_no = np.array([s.no-1 for s in Master_File[-1]])[index]
    is_multiple = np.array([np.clip(s.no-1,0,1) for s in Master_File[-1]])[index]   
    finite_vals = np.isfinite(xvals); xvals=xvals[finite_vals]; companion_no=companion_no[finite_vals]; is_multiple=is_multiple[finite_vals]
    if edges is None: #if bins are not specified
        edges = advanced_binning(xvals,np.ptp(xvals)/(len(xvals)/adaptive_no), min_bin_pop=5,allow_empty_bins=False)
    bins = (edges[1:]+edges[:-1])/2 
    if target_mass is None: #In case there's no target mass
        target_mass = (upper_limit+lower_limit)/2
    #Get MF/CF
    multip_in_bin,_,_ = stats.binned_statistic(xvals,is_multiple,statistic='sum',bins = edges)
    comp_in_bin,_,_ = stats.binned_statistic(xvals,companion_no,statistic='sum',bins = edges)
    sys_in_bin,_,_ = stats.binned_statistic(xvals,companion_no,statistic='count',bins = edges)
    MF = multip_in_bin/sys_in_bin
    CF = comp_in_bin/sys_in_bin
    #Get MF/CF error
    MF_error = np.zeros_like(MF);  CF_error = np.zeros_like(CF)
    for i in range(len(MF)):
        MF_error[i] = Psigma(sys_in_bin[i],multip_in_bin[i])
        CF_error[i] = Lsigma(sys_in_bin[i],comp_in_bin[i])
    if multiplicity == 'MF':
        y_label = 'Multiplicity Fraction'
        ylim=[0,1]
        plot_y = MF; plot_y_err = MF_error
    elif multiplicity == 'CF':
        y_label = 'Companion Frequency'
        ylim=[0,np.clip(np.max(CF+CF_error)+0.1,1,3)]
        plot_y = CF; plot_y_err = CF_error
    if plot == True:
        #Plotting the multiplicity over age
        plt.figure(figsize = (6,6))
        plt.plot(bins,plot_y, color=color)
        plt.fill_between(bins,plot_y+plot_y_err,plot_y-plot_y_err,alpha = 0.3, color=color)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim(ylim)      
        #Observations
        obs_upper = None; obs_lower = None; 
        if target_mass == 1:
            if multiplicity == 'MF':
                obs_upper = 0.46; obs_lower = 0.42;
            elif multiplicity == 'CF':
                obs_upper = 0.54; obs_lower = 0.46;
        elif target_mass == 10:
            if multiplicity == 'MF':
                obs_upper = 0.8; obs_lower = 0.4;
            elif multiplicity == 'CF':
                obs_upper = 1.8; obs_lower = 1.4;
        if not (obs_upper is None):
            xlim = plt.xlim()
            plt.fill_between(plt.xlim(),obs_upper,obs_lower,alpha = 0.3, color='k',label = 'Observations')
            plt.xlim(xlim)
        plt.text(0.01,0.01,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ r' $\mathrm{M_\odot}$',transform = plt.gca().transAxes,horizontalalignment = 'left',fontsize=14)
        plt.legend(fontsize = 14)
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
    return bins, plot_y,plot_y_err

def multiplicity_vs_formation_multi(Files,Systems,Filenames,adaptive_no = [20],T_list = None,dt_list = None,upper_limit=1.3,lower_limit = 0.7,target_mass = None,zero = 'Formation',multiplicity = 'MF',min_time_bin = 0.2,adaptive_binning = True,x_axis = 'density',labels=None, colors=None):
    '''
    The average multiplicity vs formation time/density for multiple files.

    Inputs
    ----------
    Files: list of list of sinkdata objects
    The original files before system assignment.

    Systems: list of list of star system objects
    All of the systems for the original files.
    
    Filenames: list of strings
    The names of all the files

    Parameters
    ----------
    adaptive_no: list,optional
    The number of stars in each bin for each file
    
    T_list : list,optional
    The time that the stars are born at.

    dt_list :list,optional
    The tolerance of the birth time.

    upper_limit: int,float,optional
    The upper limit of the target mass range

    lower_limit: int,float,optional
    The lower limit of the target mass range
    
    target_mass: int,float,optional
    The target mass of primary to look at
    
    zero: string,optional
    Whether to take the zero point as when the star was formed or stopped accreting. Use 'Formation' or 'Consistent Mass'.

    multiplicity: bool,optional
    Whether to plot for the multiplicity properties, multiplicity fraction or Companion Frequency.
    
    min_time_bin: int,optional
    The minimum time bin to plot on the time histogram
    
    adaptive_binning: bool,optional
    Whether to adopt adaptive binning (same no of stars in each bin)
    
    x_axis: string,optional
    Whether to plot the MF/CF with the formation time/density/mass density

    Example
    -------
    multiplicity_vs_formation(Files,Systems,adaptive_binning = True,adaptive_no = [20,20])
    '''
    if colors is None:
        colors, _ = set_colors_and_styles(None, None, len(Filenames),  dark=True)
    if labels is None: labels=Filenames
    adaptive_no = adaptive_no*len(Files)
    x_array = []
    final_mul_list = []
    yerrs = []
    for i in tqdm(range(len(Files)),position = 0):
        x,final_mul,yerr = multiplicity_vs_formation(Files[i],Systems[i],upper_limit=upper_limit,lower_limit=lower_limit,target_mass=target_mass,zero=zero,multiplicity=multiplicity,min_time_bin=min_time_bin,adaptive_no=adaptive_no[i],x_axis=x_axis,plot = True,label=labels[i], color=colors[i])
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
        plt.savefig(x_axis+'_'+multiplicity+'_'+str(i)+'.png', dpi=150) #let's just save these for now
        plt.close()
        x_array.append(x);final_mul_list.append(final_mul);yerrs.append(yerr);
    if x_axis == 'time':
        x_label = 'Formation Time[Myr]'
    elif x_axis == 'density':
        x_label = r'Log Formation Density [$\mathrm{pc}^{-3}$]'
    elif x_axis == 'mass density':
        x_label = r'Log Formation Density [$\mathrm{M_\odot}pc^{-3}$]'
    plt.figure(figsize = (6,6))
    for i in range(len(Files)):
        if not ( (labels[i]=='') or (labels[min(i+1,len(Files)-1)]=='')):
            plt.fill_between(x_array[i],final_mul_list[i]+yerrs[i],final_mul_list[i]-yerrs[i],alpha = 0.3, color=colors[i])
        plt.plot(x_array[i],final_mul_list[i], color=colors[i],label = labels[i])
    plt.text(0.01,0.01,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ r' $\mathrm{M_\odot}$',transform = plt.gca().transAxes,horizontalalignment = 'left',fontsize=14)
    plt.legend(fontsize=14)
    plt.xlabel(x_label)
    if multiplicity == 'MF':
        plt.ylabel('Multiplicity Fraction')
        plt.ylim([0,1])
    elif multiplicity == 'CF':
        plt.ylabel('Companion Frequency')
        plt.ylim([0,np.clip(plt.ylim()[1]+0.2,1,3)])
    adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)

def multiplicity_and_age_combined(file,Master_File,T_list = None,dt_list = None,upper_limit=1.3,lower_limit = 0.7,target_mass = None,zero = 'Formation',multiplicity = 'MF',filename = None,min_time_bin = 0.2,rolling_avg = False,rolling_window_Myr = 0.1,adaptive_binning = True,adaptive_no = 20,description = None, label=None):
    '''
    The average multiplicity of stars born in certain time ranges tracked throughout their lifetime in the simulation.

    Inputs
    ----------
    file: list of sinkdata objects
    The original file before system assignment.

    Master_File: list of list of star system objects
    All of the systems for the original file.

    Parameters
    ----------
    T : list,optional
    The time that the stars are born at.

    dt :list,optional
    The tolerance of the birth time.

    target_mass: int,float,optional
    The target mass of primary to look at

    upper_limit: int,float,optional
    The upper limit of the target mass range

    lower_limit: int,float,optional
    The lower limit of the target mass range

    read_in_result: bool,optional
    Whether to perform system assignment or use the already assigned system.

    select_by_time: bool,optional:
    Whether to track all stars or only those in a time frame.

    zero: string,optional
    Whether to take the zero point as when the star was formed or stopped accreting. Use 'Formation' or 'Consistent Mass'.

    plot: bool,optional
    Whether to return the times and multiplicities or plot them.

    steps: int,optional
    The number of snapshots in one bin. If reading by result, this defaults to looking at every snapshot.

    rolling_avg:bool,optional
    Whether to use a rolling average
    
    rolling_window_Myr:int,float,optional
    How much time to include in the rolling average window.
    
    adaptive_binning: bool,optional
    Whether to adopt adaptive binning (same no of stars in each bin)
    
    adaptive_no: int,optional
    The number of stars in each bin
    
    description: string,optional
    What to save the name of the Multiplicity Tracker plot under.
    
    Returns
    -------
    age_bins: array
    The age over which the stars are in.

    multiplicity: array
    The average multiplicity fraction of the objects in the bins.

    Example
    -------
    multiplicity_and_age_combined(M2e4_C_M_J_2e7,M2e4_C_M_J_2e7_systems,rolling_avg = True,adaptive_binning = True,adaptive_no = 20)
    '''  
    if label is None: label = path.basename(filename)
    #In case there's no target mass
    if description is not None:
        save = True
        output_dir = description
        new_file = output_dir+'/Multiplicity_Lifetime_Evolution'
        mkdir_p(new_file)
    else:
        save = False
    if target_mass is None:
        target_mass = (upper_limit+lower_limit)/2
    if adaptive_binning is True:
        form_times = formation_time_histogram(file,Master_File,upper_limit=upper_limit,lower_limit=lower_limit,filename=filename,only_primaries_and_singles=True,plot = False,full_form_times=True,label=label)
        form_times = np.sort(form_times)
        indices = np.array(range(0,len(form_times)-5,adaptive_no))
        adaptive_times = form_times[indices]; adaptive_times[-1] = max(adaptive_times[-2]+min_time_bin, np.max(form_times)-1.0 )
        T_list = (adaptive_times[1:]+adaptive_times[:-1])/2
        dt_list =  (adaptive_times[1:]-adaptive_times[:-1])
    time_list = []
    MF_list = []
    CF_list = []
    kept_list = []
    dens_list = []
    mass_dens_list = []
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    rolling_window = int((int(time_to_snaps(rolling_window_Myr,file))//2)*2+1)
    for i in range(len(T_list)):
        time,mul,birth_times,kept,average_dens,average_mass_dens,comp_count,all_count = MFCF_and_age(file,Master_File,T_list[i],dt_list[i],zero = zero,upper_limit=upper_limit,lower_limit = lower_limit,target_mass = target_mass,plot = False)
        time_list.append(time)
        MF_list.append(mul[0])
        CF_list.append(mul[1])
        kept_list.append(kept)
        dens_list.append(average_dens)
        mass_dens_list.append(average_mass_dens)
    #Creating a plot of formation times
    times = [s.t*code_time_to_Myr for s in file]
    birth_times = np.array(birth_times)
    if min(dt_list)<min_time_bin:
        min_time_bin = min(dt_list)
    if min_time_bin < (file[-1].t-file[-2].t)*code_time_to_Myr:
        min_time_bin = len(times)
    times,new_stars_co = hist(birth_times,bins = np.linspace(min(times),max(times),num = int((max(times)-min(times))/min_time_bin)))
    times = np.array(times)
    new_stars_co = np.insert(new_stars_co,0,0)
    plt.step(times,new_stars_co)
    ylim = plt.ylim()
    for i in range(len(T_list)):
        plt.fill_between([T_list[i]-dt_list[i]/2,T_list[i]+dt_list[i]/2],ylim[0],ylim[1],alpha  = 0.3,label = '%g Myr'%(round(T_list[i],1)))
    plt.legend(fontsize=14)
    # if filename is not None:
    #     plt.text(max(times)/2,max(new_stars_co),path.basename(filename))
    plt.text(0.02,0.9,'Primary Mass = '+str(target_mass)+' $\mathrm{M_\odot}$',transform = plt.gca().transAxes)
    plt.xlabel('Time [Myr]')
    plt.ylabel('Number of New Stars')
    adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
    if save is True:
        if filename is None:
            print('Please provide filename')
            return
        plt.savefig(new_file+'/'+str(path.basename(filename))+'/New_Stars_Histogram.png',dpi = 150, bbox_inches='tight')
    #Creating the plot of stellar densities
    number_densities = []
    mass_densities = []
    times = []
    for i in range(len(Master_File[-1])):
        if lower_limit<=Master_File[-1][i].primary<=upper_limit:
            formation_time = Master_File[-1][i].formation_time_Myr[0]
            number_density = np.log10(Master_File[-1][i].init_star_vol_density[0])
            mass_density = np.log10(Master_File[-1][i].init_star_mass_density[0])
            number_densities.append(number_density);times.append(formation_time);mass_densities.append(mass_density)
    #the_times,the_number_densities,the_errors_up,the_errors_down = density_evolution(number_densities,times,filename = filename,plot = False)
    #the_times,the_mass_densities,the_mass_errors_up,the_mass_errors_down = density_evolution(mass_densities,times,filename = filename,plot = False)
    #Number density Plots
    plt.figure(figsize = (6,6))
    density_evolution(number_densities,times,filename = filename,density= 'number',bins=np.linspace(np.percentile(form_times,1),np.percentile(form_times,99),num=9))
    # for i in range(len(T_list)):
    #     plt.fill_between([T_list[i]-dt_list[i]/2,T_list[i]+dt_list[i]/2],0,np.log10(max(the_number_densities)),alpha  = 0.3,label = 'T = '+str(round(T_list[i],2))+', dt = '+str(round(dt_list[i],2)))
    # plt.legend(fontsize=14)
    adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
    if save is True:
        if filename is None:
            print('Please provide filename')
            return
        plt.savefig(new_file+'/'+str(path.basename(filename))+'/Density_Evolution.png',dpi = 150, bbox_inches='tight')
    #Mass Density Plots
    plt.figure(figsize = (6,6))
    density_evolution(mass_densities,times,filename = filename,density = 'mass',bins=np.linspace(np.percentile(form_times,1),np.percentile(form_times,99),num=9))
    # for i in range(len(T_list)):
    #     plt.fill_between([T_list[i]-dt_list[i]/2,T_list[i]+dt_list[i]/2],0,np.log10(max(the_mass_densities)),alpha  = 0.3,label = 'T = '+str(round(T_list[i],2))+', dt = '+str(round(dt_list[i],2)))
    # plt.legend(fontsize=14)
    adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
    #Using rolling average
    if rolling_avg is True:
        for i in range(len(time_list)):
            #time_list[i] = np.array(rolling_average(time_list[i],rolling_window))
            MF_list[i] = np.array(rolling_average(MF_list[i],rolling_window))
            CF_list[i] = np.array(rolling_average(CF_list[i],rolling_window))
    if save is True:
        if filename is None:
            print('Please provide filename')
            return
        plt.savefig(new_file+'/'+str(path.basename(filename))+'/Mass_Density_Evolution.png',dpi = 150, bbox_inches='tight')
    #Plotting the MF over age
    if multiplicity == 'MF' or multiplicity == 'both':
        plt.figure(figsize = (6,6))
        mul_list = MF_list
        for i in range(len(time_list)):
            xvals = np.array(time_list[i])
            index = xvals<(max(times)-T_list[i]-dt_list[i]/2) # conservative cut to keep things representative
            plt.plot(xvals[index],np.array(mul_list[i])[index],label =  '%g Myr'%(round(T_list[i],1)))
        if target_mass == 1:
            plt.errorbar(max(list(flatten(time_list)))*0.8,0.44,yerr=0.02,marker = 'o',capsize = 5,color = 'black',label = 'Observations',ls='none')
        elif target_mass == 10:
            plt.errorbar(max(list(flatten(time_list)))*0.8,0.6,yerr=0.2,lolims = True,marker = 'o',capsize = 5,color = 'black',label = 'Observations',ls='none')
        plt.legend(fontsize=14)
        plt.xlabel('Age [Myr]')
        plt.ylabel('Multiplicity Fraction')
        plt.ylim([-0.05,1.05])
        plt.text(max(list(flatten(time_list)))/2*0.7,0.1,'Primary Mass = '+str(target_mass)+' $\mathrm{M_\odot}$')
        # if filename is not None:
        #     plt.text(max(list(flatten(time_list)))/2,0.5,label)
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
        if save is True:
            if filename is None:
                print('Please provide filename')
                return
            plt.savefig(new_file+'/'+path.basename(str(filename))+'/Multiplicity_Fraction_Lifetime_Evolution.png',dpi = 150, bbox_inches='tight')
    #Plotting the CF over age
    if multiplicity == 'CF' or multiplicity == 'both':
        plt.figure(figsize = (6,6))
        mul_list = CF_list
        for i in range(len(time_list)):
            xvals = np.array(time_list[i])
            index = xvals<(max(times)-T_list[i]-dt_list[i]/2) # conservative cut to keep things representative
            plt.plot(xvals[index],np.array(mul_list[i])[index],label =  '%g Myr'%(round(T_list[i],1)))
        if target_mass == 10:
             plt.errorbar(max(list(flatten(time_list)))*0.8,1.6,yerr=0.2,lolims = True,marker = 'o',capsize = 5,color = 'black',label = 'Observations',ls='none')
        elif target_mass == 1:
            plt.errorbar(max(list(flatten(time_list)))*0.8,0.5,yerr=0.04,marker = 'o',capsize = 5,color = 'black',label = 'Observations',ls='none')
        plt.legend(fontsize=14)
        plt.xlabel('Age [Myr]')
        plt.ylabel('Companion Frequency')
        plt.ylim([-0.05,max(list(flatten(CF_list)))])
        plt.text(max(list(flatten(time_list)))/2*0.7,0.1,'Primary Mass = '+str(target_mass)+' $\mathrm{M_\odot}$',fontsize=14)
        # if filename is not None:
        #     plt.text(max(list(flatten(time_list)))/2,0.5,label)
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
        if save is True:
            if filename is None:
                print('Please provide filename')
                return
            plt.savefig(new_file+'/'+path.basename(str(filename))+'/Companion_Frequency_Lifetime_Evolution.png',dpi = 150, bbox_inches='tight')

def One_Snap_Plots(which_plot,Master_File,file,systems = None,filename = None,snapshot = -1,upper_limit = 1.3,lower_limit = 0.7,target_mass = None,all_companions = True,bins = 10,log = True,compare = False,plot = True,read_in_result = True,filters = ['q_filter','time_filter'],avg_filter_snaps_no = 10,q_filt_min = 0.1,time_filt_min = 0.1,only_filter = True,label=None,filter_in_class = True, plot_intermediate_filters = False):
    '''
    Create the plots for one snapshot
    Inputs
    ----------
    which_plot: string
    The plot to be made.
    
    Master_File: list of lists of star system objects
    The entire simulation with system assignment.

    file:list of sinkdata objects
    The original sinkdata file.

    Parameters
    ----------
    systems: list of star system objects
    The systems you want to analyze (1 snapshot or a filtered snapshot).
    
    filename:string
    The name of the original file. It will be labelled on the plot if provided.

    snapshot: int,float
    The snapshot number you want to look at. Only required for IMF comparisons and filter on.

    target_mass: int,float,optional
    The mass of the primaries of the systems of interest.

    upper_limit: int,float,optional
    The upper mass limit of the primaries of systems of interest.

    lower_limit: int,float,optional
    The lower mass limit of the primaries of systems of interest.

    all_companions: bool,optional
    Whether to include all companions in the mass ratio or semi major axes.

    bins: int,float,list,array,string,optional
    The bins for the histograms.

    log: bool,optional
    Whether to plot the y data on a log scale.

    plot: bool,optional
    Whether to plot the data or just return it.

    read_in_result: bool,optional
    Whether to perform system assignment again or just read it in.

    filters: list of strings,None,optional
    Whether to use a filter or not. The choices are 'time_filter','q_filter' or 'average_filter'
    
    avg_filter_snaps_no: int,optional
    The number of snapshots to average over with the average filter
    
    q_filt_min: float,optional
    The minimum q to use in the q filter
    
    time_filt_min: float,optional
    The minimum time that companions should have been in a system
     
    only_filter: bool,optional
    Whether to only look at the filter (True) or to plot both the filter and unfiltered data (False).
    
    label: string,optional
    The label to use on the plot
    
    filter_in_class: bool,optional
    If the filter is saved in the class definition.
    
    Returns
    -------
    x_vals: list
    The bins.

    weights:list
    The weights of each bin

    NOTE: See Plots documentation for a better description.

    Example
    -------
    One_Snap_Plots('Mass Ratio',M2e4_C_M_J_2e7_systems[-1],M2e4_C_M_J_2e7)
    '''
    adjust_ticks=True
    if label is None: label=filename
    if snapshot<0: snapshot = len(Master_File)+snapshot
    if systems is None: systems = Master_File[snapshot]
    filtered_systems = []; filters_done = []; filters_to_plot = []
    #linestyle_list = [':','--','-.','-']
    linestyle_list = ['-','-','-','-','-','-']
    if not (filters[0] is None or filters[0]=='None'):
        # if 'time_filter' in filters and 'q_filter' in filters and filter_in_class is True:
        #     filtered_systems = [get_q_and_time(filtered_systems)]
        # else:
        if 'time_filter' in filters:
            filtered_systems.append(full_simple_filter(Master_File,file,snapshot,long_ago = time_filt_min,filter_in_class=filter_in_class))
            filters_done.append(r'$t_\mathrm{comp}$>%3.2g Myr'%(time_filt_min))
        if 'q_filter' in filters:
            filtered_systems.append(q_filter_one_snap(systems,min_q=q_filt_min,filter_in_class=filter_in_class))
            filters_done.append(r'q>%3.2g'%(q_filt_min))
        if 'Raghavan' in filters:
            filtered_systems.append(Raghavan_filter_one_snap(systems))
            filters_done.append(r'MDS17 correction')
        if ('time_filter' in filters) and ( ('q_filter' in filters) or ('Raghavan' in filters) ):
            filtered_systems.append(full_simple_filter(Master_File[:snapshot]+[filtered_systems[-1]]+Master_File[(snapshot+1):],file,snapshot,long_ago = time_filt_min,filter_in_class=filter_in_class))
            #filtered_systems.append(q_filter_one_snap(filtered_systems[-1],min_q=q_filt_min,filter_in_class=filter_in_class))
            #filters_done.append(filters_done[-2]+' & '+filters_done[-1])
            filters_done.append('All corrections')
        if only_filter is True:
            systems = filtered_systems[-1]    
        else:
            if plot_intermediate_filters: 
                filters_to_plot = filters_done
            else:
                filters_to_plot = [filters_done[-1]]
        colorlist, _ = set_colors_and_styles(None, None, len(filters_to_plot)+1, dark=True, sequential=True)
    property_dist = primary_total_ratio_axis(systems,lower_limit=lower_limit,upper_limit=upper_limit,all_companions=all_companions,attribute=which_plot,file = file)
    if which_plot == 'Mass Ratio' or which_plot == 'Eccentricity':
        if bins is None:
            if len(property_dist)/5 < 10:
                noofbins = int(len(property_dist)/5)
            else:
                noofbins = 10
            bins = np.linspace(0,1,noofbins+1)
        transform_func = lambda x : x
    elif which_plot == 'Semi Major Axis':
        if bins is None:
            data_array = np.log10(property_dist)-np.log10(m_to_AU)
            floor = np.floor(np.min(data_array[~np.isnan(data_array)]))
            ceiling = np.ceil(np.max(data_array[~np.isnan(data_array)]))      
            bins = np.linspace(floor,ceiling,int((ceiling-floor)*(3/2)+1))
        transform_func = lambda x : np.log10(x)-np.log10(m_to_AU)
    elif which_plot == 'Angle':
        if bins is None:
            bins = np.linspace(0,180,10)
        transform_func = lambda x : x
    else:
        if bins is None:
            data_array = np.log10(property_dist)
            floor = np.floor(np.min(data_array[~np.isnan(data_array)]))
            ceiling = np.ceil(np.max(data_array[~np.isnan(data_array)]))      
            bins = np.linspace(floor,ceiling,int((ceiling-floor)*2+1))
        transform_func = lambda x : np.log10(x)
    x_vals,y_vals = hist(transform_func(property_dist),bins = bins)
    y_vals = np.insert(y_vals,0,0)
    #Creating data for the filtered systems
    x_vals_filt = {};  y_vals_filt = {};
    for f_systems, filter_label in zip(filtered_systems, filters_done):
        property_dist_filt = primary_total_ratio_axis(f_systems,lower_limit=lower_limit,upper_limit=upper_limit,all_companions=all_companions,attribute=which_plot,file = file)
        x_vals_filt[filter_label], y_vals_filt[filter_label] = hist(transform_func(property_dist_filt),bins = bins)
        y_vals_filt[filter_label]=np.insert(y_vals_filt[filter_label],0,0)
    
    if which_plot == 'System Mass' or which_plot == 'Primary Mass':
        if plot == True:
            #plt.title('Total Mass Distribution of all of the systems in Snapshot '+str(snapshot_number))
            if which_plot == 'System Mass':
                plt.xlabel('Log System Mass [$\mathrm{M_\odot}$]')
            else:
                plt.xlabel('Log Primary Mass [$\mathrm{M_\odot}$]')
            plt.ylabel('Number of systems')
            if compare == True: #If we want to compare the total mass function to the system mass function
                if snapshot is None:
                    print('please provide snapshot')
                    return
                tot_m,vals = hist(np.log10(file[snapshot].m),bins = bins)
                vals = np.insert(vals,0,0)
                vals = vals*sum(y_vals)/sum(vals)
                plt.xlabel('Log Mass[$\mathrm{M_\odot}$]')
                if which_plot == 'System Mass':
                    plt.step(x_vals,y_vals,label = 'Systems')
                else:
                    plt.step(x_vals,y_vals,label = 'Primary stars')
                    plt.ylabel('Number of stars')
                plt.step(tot_m+0.01,vals,label = 'All stars (IMF)',linestyle='--')
                plt.legend(fontsize=14)
            elif only_filter is False:
                plt.step(x_vals,y_vals,label = 'Simulation', color = colorlist[0])
                #plt.step(x_vals_filt-0.01,y_vals_filt-0.1,label = 'After Corrections',linestyle = ':')
                for i, (filter_label,linestyle,linecolor) in enumerate(zip(filters_to_plot,linestyle_list,colorlist[1:])):
                    plt.step(x_vals_filt[filter_label]-0.01*(i+1),y_vals_filt[filter_label]+0.05*(i+1),label = filter_label, linestyle = linestyle,color=linecolor)
                plt.legend(fontsize=14)
            else:
                plt.step(x_vals,y_vals)
            if log == True:
                plt.yscale('log')
            # if filename is not None:
            #     plt.text(0.7,0.7,label,transform = plt.gca().transAxes,horizontalalignment = 'left')
            # plt.text(0.7,0.3,'Total Number of systems ='+str(sum(y_vals)),transform = plt.gca().transAxes,fontsize = 14,horizontalalignment = 'left')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        else:
            return x_vals,y_vals
    if which_plot == 'Mass Ratio':
        if plot == True:
            plt.step(x_vals,y_vals,label = 'Simulation',color=colorlist[0])
            if only_filter is False:
                # plt.step(x_vals_filt-0.01,y_vals_filt-0.1,label = 'After Corrections',linestyle = ':')
                for i, (filter_label,linestyle,linecolor) in enumerate(zip(filters_to_plot,linestyle_list,colorlist[1:])):
                    plt.step(x_vals_filt[filter_label]-0.01*(i+1),y_vals_filt[filter_label]+0.05*(i+1),label = filter_label, linestyle = linestyle,color=linecolor)
            plt.ylabel('Number of systems')
            plt.xlabel(r'q ($M/M_\mathrm{p}$)')
            # if filename is not None:
            #     plt.text(0.5,0.7,label,transform = plt.gca().transAxes,fontsize=14,horizontalalignment = 'left')  
            #Raghavan 2010 mass ratio data for Solar type binaries
            x_obs = np.linspace(0.05,0.95,10)
            x_obs_errs = 0.05*np.ones_like(x_obs)
            y_obs = np.array([4,6,12,16,11,12,12,9,11,17]) #for binaries only
            y_obs_err = np.sqrt(y_obs) #Poisson error
            obs_norm_factor = np.mean(y_vals)/np.mean(y_obs)*0.9
            if len(filters_to_plot)>1:
                obs_norm_factor = np.mean(y_vals_filt[filters_to_plot[-1]])/np.mean(y_obs)
            plt.errorbar(x_obs-0.01,y_obs*obs_norm_factor,yerr=y_obs_err*obs_norm_factor,xerr = x_obs_errs,marker = 'o',capsize = 5,color = 'black',label = 'Raghavan 2010',ls='none')           
            if upper_limit<1000:
                plt.text(0.98,0.65,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ ' $\mathrm{M_\odot}$',transform = plt.gca().transAxes,horizontalalignment = 'right', fontsize=14)
            if compare == True:
                if snapshot is None:
                    print('Please provide snapshots')
                    return
                if target_mass is None:
                    print('Please provide target_mass')
                    return
                Weighted_IMF,IMF = randomly_distributed_companions(systems,file,snapshot,mass_ratio=bins,plot = False,upper_limit=upper_limit,lower_limit=lower_limit,target_mass=target_mass)
                IMF = np.insert(IMF,0,0)
                if len(x_vals)>4:
                    plt.vlines((x_vals[-1]+x_vals[-2])/2,y_vals[-1]-np.sqrt(y_vals[-1]),y_vals[-1]+np.sqrt(y_vals[-1]),alpha = 1.0,color=colorlist[0])
                    plt.vlines((x_vals[4]+x_vals[3])/2,y_vals[4]-np.sqrt(y_vals[4]),y_vals[4]+np.sqrt(y_vals[4]),alpha = 1.0,color=colorlist[0])
                    plt.vlines((x_vals[1]+x_vals[2])/2,y_vals[2]-np.sqrt(y_vals[2]),y_vals[2]+np.sqrt(y_vals[2]),alpha = 1.0,color=colorlist[0])
                    if only_filter is False:
                        plt.vlines((x_vals_filt[filters_to_plot[-1]][-1]+x_vals_filt[filters_to_plot[-1]][-2]-0.02)/2,y_vals_filt[filters_to_plot[-1]][-1]-np.sqrt(y_vals_filt[filters_to_plot[-1]][-1]),y_vals_filt[filters_to_plot[-1]][-1]+np.sqrt(y_vals_filt[filters_to_plot[-1]][-1]),color = colorlist[-1])
                        plt.vlines((x_vals_filt[filters_to_plot[-1]][4]+x_vals_filt[filters_to_plot[-1]][3]-0.02)/2,y_vals_filt[filters_to_plot[-1]][4]-np.sqrt(y_vals_filt[filters_to_plot[-1]][4]),y_vals_filt[filters_to_plot[-1]][4]+np.sqrt(y_vals_filt[filters_to_plot[-1]][4]),color = colorlist[-1])
                        plt.vlines((x_vals_filt[filters_to_plot[-1]][1]+x_vals_filt[filters_to_plot[-1]][2]-0.02)/2,y_vals_filt[filters_to_plot[-1]][2]-np.sqrt(y_vals_filt[filters_to_plot[-1]][2]),y_vals_filt[filters_to_plot[-1]][2]+np.sqrt(y_vals_filt[filters_to_plot[-1]][2]),color = colorlist[-1])
                plt.step(x_vals+0.01,(IMF*sum(y_vals)/sum(IMF))+0.01,label = 'All stars (IMF)', linestyle='--')
                if all_companions == True:
                    plt.ylabel('Number of stars')
                else:
                    plt.step(x_vals+0.01,(Weighted_IMF*sum(y_vals)/sum(Weighted_IMF))+0.01,label = 'Weighted IMF')
            plt.legend(fontsize=14,labelspacing=0)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            if log == True:
                plt.yscale('log')
            m_prim_guess = (lower_limit + upper_limit)/2
            xlims = list(plt.xlim()); ylims = list(plt.ylim()); xlims[0] = max(xlims[0],-0.05); xlims[1] = min(xlims[1],1.05)
            plt.fill_between(np.linspace(xlims[0],0.1/m_prim_guess,100),ylims[1],ylims[0],color = 'black',alpha = 0.1)
            #plt.text(0,np.mean(ylims),'Incomplete',horizontalalignment = 'left',fontsize=14,verticalalignment = 'top', rotation='vertical')
            plt.xlim(xlims); plt.ylim(ylims);
        else:
            return x_vals,y_vals
    if which_plot == 'Eccentricity':
        if plot == True:
            plt.step(x_vals,y_vals,label = 'Simulation', color=colorlist[0])
            if only_filter is False:
                # plt.step(x_vals_filt-0.01,y_vals_filt-0.1,label = 'After Corrections',linestyle = ':')
                for i, (filter_label,linestyle,linecolor) in enumerate(zip(filters_to_plot,linestyle_list,colorlist[1:])):
                    plt.step(x_vals_filt[filter_label]-0.01*(i+1)*np.ptp(plt.xlim()),y_vals_filt[filter_label]+0.01*(i+1)*np.ptp(plt.ylim()),label = filter_label, linestyle = linestyle, color = linecolor)
            # #Raghavan 2010 eccentricity data
            # x_obs = np.linspace(0.05,0.95,10)
            # x_obs_errs = 0.05*np.ones_like(x_obs)
            # y_obs1 = np.array([4,12,11,14,14,16,10,11,4,5])*82/100 #for log P<3
            # y_obs2 = np.array([12,15,15,18,6,21,6,3,3,3])*35/100 #for log P<3
            # y_obs = y_obs1+y_obs2 #Let's take all
            # y_obs_err = np.sqrt(y_obs) #Poisson error
            # obs_label = 'Raghavan 2010'
            #Tokovinin 2016 eccentricity data
            x_obs = np.linspace(0.1,0.9,5)
            x_obs_errs = 0.1*np.ones_like(x_obs)
            y_obs = np.array([0.08,0.17,0.25,0.28,0.25])   
            y_obs_err = 0.05*np.ones_like(y_obs)
            obs_label = 'Tokovinin+2016'
            obs_norm_factor = np.mean(y_vals)/np.mean(y_obs)
            plt.errorbar(x_obs+0.01,y_obs*obs_norm_factor,yerr=y_obs_err*obs_norm_factor,xerr = x_obs_errs,marker = 'o',capsize = 5,color = 'black',label = obs_label,ls='none')
            plt.ylabel('Number of systems')
            plt.xlabel('Eccentricity') 
            if upper_limit<1000:
                plt.text(0.02,0.02,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ ' $\mathrm{M_\odot}$',transform = plt.gca().transAxes,horizontalalignment = 'left',fontsize=14,verticalalignment = 'bottom')
            plt.legend(fontsize=14,labelspacing=0,loc=2)
            if log == True:
                plt.yscale('log')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        else:
            return x_vals,y_vals
    
    
    
    if which_plot == 'Angle':
        if plot == True:
            plt.step(x_vals,y_vals,label = 'Simulation', color=colorlist[0])
            if only_filter is False:
                # plt.step(x_vals_filt-0.01,y_vals_filt-0.1,label = 'After Corrections',linestyle = ':')
                for i, (filter_label,linestyle,linecolor) in enumerate(zip(filters_to_plot,linestyle_list,colorlist[1:])):
                    plt.step(x_vals_filt[filter_label]-0.01*(i+1)*np.ptp(plt.xlim()),y_vals_filt[filter_label]+0.01*(i+1)*np.ptp(plt.ylim()),label = filter_label, linestyle = linestyle, color = linecolor)
            plt.ylabel('Number of systems')
            plt.xlabel('Misalignment Angle [°]')
            x_pected = np.linspace(0,180,31)
            plt.plot(x_pected,max(y_vals)*np.sin(x_pected*np.pi/180),linestyle = '--',label = 'Random Alignment')
            # if filename is not None:
            #     plt.text(0.5,0.7,label,transform = plt.gca().transAxes,fontsize = 14,horizontalalignment = 'left')  
            if upper_limit<1000:
                plt.text(0.02,0.95,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ ' $\mathrm{M_\odot}$',transform = plt.gca().transAxes,horizontalalignment = 'left',fontsize=14)
            plt.legend(fontsize=14,labelspacing=0,loc=3)
            if log == True:
                plt.yscale('log')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        else:
            return x_vals,y_vals
    if which_plot == 'Semi Major Axis':
        if plot == True:
            fig = plt.figure(figsize = (6,6))
            ax1 = fig.add_subplot(111)
            ax1.step(x_vals,y_vals,label = 'Simulation', color=colorlist[0])
            if only_filter is False:
                #ax1.step(x_vals_filt-0.01,y_vals_filt-0.1,label = 'After Corrections',linestyle = ':')
                for i, (filter_label,linestyle,linecolor) in enumerate(zip(filters_to_plot,linestyle_list,colorlist[1:])):
                    ax1.step(x_vals_filt[filter_label]-0.05*(i+1),y_vals_filt[filter_label]+0.1*(i+1),label = filter_label, linestyle = linestyle, color = linecolor)
            ax1.vlines(np.log10(20),0,max(y_vals),color='k',linestyle=':')
            pands = []
            for i in systems:
                if i.no>1 and lower_limit<=i.primary<=upper_limit:
                    pands.append(i.primary+i.secondary)
            average_pands = np.average(pands)*1.9891e30 
            ax1.set_ylim([0,max(y_vals)+1])
            ax1.set_xlabel('Log Semi Major Axis [AU]')
            plt.ylabel('Number of systems')
            ax2 = ax1.twiny()
            adjust_ticks=False
            ax2.set_xlabel('Log Period [Days]')
            logperiod_lims = np.log10(2*np.pi*np.sqrt(((10**np.array(ax1.get_xlim())*m_to_AU)**3)/(6.67e-11*average_pands))/(s_to_day))
            ax2.set_xlim(logperiod_lims)
            if (upper_limit == 1.3 and lower_limit == 0.7) or (lower_limit>=5.0):
                # periods = np.linspace(3.5,7.5,num = 5)
                # k = ((10**periods)*24*60*60)
                # smaxes3 = ((6.67e-11*(k**2)*average_pands)/(4*np.pi**2))
                # smaxes = np.log10((smaxes3**(1/3))/m_to_AU)
                # error_values_small = np.array([6,7,9,9,10])
                # error_values_big = np.array([18,27,31,23,21])
                # error_values_comb = (error_values_small+error_values_big)
                # dy_comb = np.sqrt(error_values_comb)
                # ax1.errorbar(smaxes,np.array(error_values_comb)*max(y_vals)/max(error_values_comb),yerr=dy_comb*max(y_vals)/max(error_values_comb),xerr = (2/3)*0.5*np.ones_like(len(smaxes)),marker = 'o',capsize = 5,color = 'black',label = 'MDS 2017',linestyle = '')
                
                #companion frequebcies across period bins
                logperiod_day_to_logsmaxes_AU =  lambda x :  np.log10((((((10.0**x)*24*60*60)**2) * 6.67e-11 * average_pands / (4*np.pi**2))**0.333)/m_to_AU)  
                if (upper_limit == 1.3 and lower_limit == 0.7): 
                    obs_periods = np.linspace(3.5,7.5,num = 5)
                    obs_y = np.array([6,7,9,9,10])+np.array([18,27,31,23,21])
                    obs_y_error = np.sqrt(obs_y)
                else:
                    obs_periods =np.array([1,3,5,7])
                    obs_y = np.array([0.16, 0.24, 0.21, 0.12])
                    obs_y_error = np.array([0.05, 0.08, 0.07, 0.04])
                obs_smaxes = logperiod_day_to_logsmaxes_AU(obs_periods) 
                #obs_norm_factor = np.interp(obs_smaxes[np.argmax(obs_y)],x_vals,y_vals)/np.max(obs_y)
                obs_norm_factor = np.max(y_vals[x_vals>np.log10(20)+0.2])/np.max(obs_y*0.98)
                ax1.errorbar(obs_smaxes-0.01,obs_y*obs_norm_factor,yerr=obs_y_error*obs_norm_factor,xerr = 0.5*np.ones_like(len(obs_smaxes)),marker = 'o',capsize = 5,color = 'black',label = 'MDS 2017',linestyle = '')
            if log == True:
                plt.yscale('log')
            ax1.set_ylabel('Number of systems')
            if all_companions == True:
                ax1.set_ylabel('Number of subsystems')
            ax1.legend(fontsize = 14,labelspacing=0, loc=1)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            if upper_limit<1000:
                fig.text(0.99,0.675,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ ' $\mathrm{M_\odot}$',transform = plt.gca().transAxes,horizontalalignment = 'right',fontsize=14)  
            # if filename is not None:
            #     fig.text(0.5,0.7,str(filename),transform = plt.gca().transAxes,horizontalalignment = 'left',fontsize=14) 
        else:
            return x_vals,y_vals
    if which_plot == 'Semi-Major Axis vs q':
        q = primary_total_ratio_axis(systems,lower_limit=lower_limit,upper_limit = upper_limit,attribute='Mass Ratio',file = file)
        smaxes = primary_total_ratio_axis(systems,lower_limit=lower_limit,upper_limit = upper_limit,attribute='Semi Major Axis',file = file)
        plt.figure(figsize= (6,6))
        #plt.title('Mass Ratio vs Semi Major Axis for a target mass of '+str(target_mass)+' in '+Files_key[systems_key])
        plt.xlabel('Semi Major Axis (in log AU)')
        plt.ylabel('Mass Ratio')
        plt.scatter(np.log10(smaxes)-np.log10(m_to_AU),q)
        # if filename is not None:
        #     plt.text(0.7,0.7,label,transform = plt.gca().transAxes,fontsize=14,horizontalalignment = 'left')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,adjust_ticks=adjust_ticks)

def Multiplicity_One_Snap_Plots(Master_File,file,systems = None,snapshot = -1,filename = None,plot = True,multiplicity = 'MF',mass_break=2,bins = 'observer',filters = ['q_filter','time_filter'],avg_filter_snaps_no = 10,q_filt_min = 0.1,time_filt_min = 0.1,only_filter = True,label=None,filter_in_class = True,style = 'line'):
    '''
    Create a plot for the multiplicity over a mass range for a single snapshot.

    Inputs
    ----------
    Master_File: list of lists of star system objects
    The entire simulation with system assignment.
    
    file: list of sinkdata
    The simulation file.

    Parameters
    ----------
    systems: list of star system objects,optional
    The systems you want to analyze (1 snapshot or a filtered snapshot).

    snapshot: int,float
    The snapshot to look at. It is required with the filter on.

    filename: string,optional
    The name of the file to look at. It will be put on the plot if provided.

    plot: bool,optional
    Whether to plot or just return the values

    multiplicity: bool,optional
    Whether to plot for the multiplicity properties, multiplicity fraction or Companion Frequency.

    mass_break:
    The spacing between masses in log space. This is used for the continous bins.

    bins: int,float,list,array,string,optional
    The bins for the histograms. Use continous or observer.

    filters: list of strings,None,optional
    Whether to use a filter or not. The choices are 'time_filter','q_filter' or 'average_filter'
    
    avg_filter_snaps_no: int,optional
    The number of snapshots to average over with the average filter
    
    q_filt_min: float,optional
    The minimum q to use in the q filter
    
    time_filt_min: float,optional
    The minimum time that companions should have been in a system
     
    only_filter: bool,optional
    Whether to only look at the filter (True) or to plot both the filter and unfiltered data (False).
    
    label: string,optional
    The label to put on the plot
    
    filter_in_class: bool,optional
    If the filter is saved in the class definition.
    
    Returns
    -------
    x_vals: list
    The bins.

    weights:list
    The weights of each bins

    NOTE: Refer to Plots documentation for a better explanation

    Returns
    -------
    logmasslist: The masses in log space.

    o1: The first output. It is the multiplicity fraction, Companion Frequency or the primary star fraction (depending on the attribute).

    o2: The second output. It is the number of multistar systems, number of companions or the single star fraction (depending on the attribute).

    o3: The third output. It is the number of all systems or the companion star fraction (depending on the attribute).

    NOTE: If filter is on, the filtered output will be returned.

    Examples
    -------
    1) Multiplicity_One_Snap_Plots(M2e4_C_M_J_2e7_systems[-1],multiplicity = 'MF',bins = 'observer')
    Simple multiplicity fraction plot.

    2) Multiplicity_One_Snap_Plots(M2e4_C_M_J_2e7_systems[-1],multiplicity = 'CF',bins = 'observer',filtered = True,snapshot = -1,Master_File = M2e4_C_M_J_2e7_systems)
    Companion Frequency Plot with filter on. 

    3) Multiplicity_One_Snap_Plots(M2e4_C_M_J_2e7_systems[-1],multiplicity = 'Properties',bins = 'observer',plot = False)
    Multiplicity properties values being returned.
    '''
    if label is None: label=filename
    if bins is None:
        bins = 'continous'
    if systems is None: systems = Master_File[snapshot]
    filtered_systems = systems
    if 'time_filter' in filters and 'q_filter' in filters and filter_in_class is True:
        filtered_systems = get_q_and_time(filtered_systems)
    else:
        if 'time_filter' in filters:
            filtered_systems = full_simple_filter(Master_File,file,snapshot,long_ago = time_filt_min,filter_in_class=filter_in_class)
        if 'q_filter' in filters:
            filtered_systems = q_filter_one_snap(filtered_systems,min_q=q_filt_min,filter_in_class=filter_in_class)
    if only_filter is True:
        systems = filtered_systems
    if multiplicity == 'CF':
        logmasslist,o1,o2,o3 = companion_frequency(systems,mass_break=mass_break,bins = bins)
    elif multiplicity == 'Properties' or multiplicity == 'MF':
        logmasslist,o1,o2,o3 = multiplicity_fraction(systems,attribute=multiplicity,mass_break=mass_break,bins = bins)
    else:
        if file is None:
            print('Provide file')
            return
        if multiplicity == 'Mass Density Separate' or multiplicity == 'Density Separate':
            logmasslist,o1,o2,o3,o4,o5,o6 = multiplicity_fraction_with_density(systems,file,mass_break=mass_break,bins = bins,attribute=multiplicity)
        else:
            logmasslist,o1,o2 = multiplicity_fraction_with_density(systems,file,mass_break=mass_break,bins = bins,attribute=multiplicity)
    if only_filter is False or 'average_filter' in filters:
        logmasslist_all = []
        o1_all = []
        o2_all = []
        o3_all = []
        count = 0
        if 'average_filter' in filters:
            for i in range(snapshot+1-avg_filter_snaps_no,snapshot+1):
                filtered_q = Master_File[i]
                if 'time_filter' in filters and 'q_filter' in filters and filter_in_class is True:
                    filtered_q = get_q_and_time(Master_File[i])
                else:
                    if 'time_filter' in filters:
                        filtered_q = full_simple_filter(Master_File,file,i,long_ago = time_filt_min,filter_in_class=filter_in_class)
                    if 'q_filter' in filters:
                        filtered_q = q_filter_one_snap(filtered_q,min_q=q_filt_min,filter_in_class=filter_in_class)
                if multiplicity == 'CF':
                    logmasslist_all.append(companion_frequency(filtered_q,mass_break=mass_break,bins = bins)[0])
                    o1_all.append(companion_frequency(filtered_q,mass_break=mass_break,bins = bins)[1])
                    o2_all.append(companion_frequency(filtered_q,mass_break=mass_break,bins = bins)[2])
                    o3_all.append(companion_frequency(filtered_q,mass_break=mass_break,bins = bins)[3])
                else:
                    logmasslist_all.append(multiplicity_fraction(filtered_q,attribute = multiplicity,mass_break=mass_break,bins = bins)[0])
                    o1_all.append(multiplicity_fraction(filtered_q,attribute = multiplicity,mass_break=mass_break,bins = bins)[1])
                    o2_all.append(multiplicity_fraction(filtered_q,attribute = multiplicity,mass_break=mass_break,bins = bins)[2])
                    o3_all.append(multiplicity_fraction(filtered_q,attribute = multiplicity,mass_break=mass_break,bins = bins)[3])
                count += 1
            if 'average_filter' in filters:
                logmasslist_filt = np.zeros_like(logmasslist_all[-1])
                o1_filt = np.zeros_like(o1_all[-1])
                o2_filt = np.zeros_like(o2_all[-1])
                o3_filt = np.zeros_like(o3_all[-1])
                count = 0
                for i in range(snapshot+1-avg_filter_snaps_no,snapshot+1):
                    for j in range(len(logmasslist_filt)):
                        logmasslist_filt[j] += logmasslist_all[count][j]
                        o1_filt[j] += o1_all[count][j]
                        o2_filt[j] += o2_all[count][j]
                        o3_filt[j] += o3_all[count][j]
                    count += 1
                logmasslist_filt = logmasslist_filt/avg_filter_snaps_no
                o1_filt = o1_filt/avg_filter_snaps_no
                o2_filt = o2_filt/avg_filter_snaps_no
                o3_filt = o3_filt/avg_filter_snaps_no
                if only_filter is True:
                    logmasslist = logmasslist_filt
                    o1 = o1_filt
                    o2 = o2_filt
                    o3 = o3_filt
        else:
            filtered_q = Master_File[snapshot]
            if 'time_filter' in filters and 'q_filter' in filters and filter_in_class is True:
                filtered_q = get_q_and_time(Master_File[snapshot])
            else:
                if 'time_filter' in filters:
                    filtered_q = full_simple_filter(Master_File,file,snapshot,long_ago = time_filt_min,filter_in_class=filter_in_class)
                if 'q_filter' in filters:
                    filtered_q = q_filter_one_snap(filtered_q,min_q=q_filt_min,filter_in_class=filter_in_class)
            if multiplicity == 'CF':
                logmasslist_filt,o1_filt,o2_filt,o3_filt = companion_frequency(filtered_q,mass_break=mass_break,bins = bins)
            else:
                logmasslist_filt,o1_filt,o2_filt,o3_filt = multiplicity_fraction(filtered_q,attribute = multiplicity,mass_break=mass_break,bins = bins)
    if multiplicity == 'Properties' or multiplicity == 'Density Separate' or multiplicity == 'Mass Density Separate':
        if plot == True:
            if multiplicity == 'Properties':
                markersize = 10; linewidth = 2.0
                plt.plot(logmasslist,o1,marker = '*',label = 'Unbound stars', markersize=markersize, linewidth=linewidth)
                plt.plot(logmasslist,o2,marker = 'o', label = 'Primary stars', markersize=markersize, linewidth=linewidth)
                plt.plot(logmasslist,o3,marker = '^',label = 'Non-primary stars', markersize=markersize, linewidth=linewidth)
            else:
                plt.plot(logmasslist,np.log10(o1),marker = '*',label = 'Unbound stars')
                plt.fill_between(logmasslist,np.log10(o1+o4),np.log10(o1)-(np.log10(o1+o4)-np.log10(o1)),alpha = 0.3)
                plt.plot(logmasslist,np.log10(o2),marker = 'o', label = 'Primary stars')
                plt.fill_between(logmasslist,np.log10(o2+o5),np.log10(o2)-(np.log10(o2+o5)-np.log10(o2)),alpha = 0.3)
                plt.plot(logmasslist,np.log10(o3),marker = '^',label = 'Non-primary stars')
                plt.fill_between(logmasslist,np.log10(o3+o6),np.log10(o3)-(np.log10(o3+o6)-np.log10(o3)),alpha = 0.3)
            if only_filter is False:
                plt.plot(logmasslist_filt,o1_filt,marker = '*',label = 'Primary stars, filtered',linestyle = ':')
                plt.plot(logmasslist_filt,o2_filt,marker = 'o', label = 'Unbound stars, filtered',linestyle = ':')
                plt.plot(logmasslist_filt,o3_filt,marker = '^',label = 'Non-primary stars, filtered',linestyle = ':')
            plt.legend(fontsize=14)
            xlims = plt.xlim(); ylims = plt.ylim()
            plt.fill_between(np.linspace(xlims[0],0,20),ylims[1],ylims[0],color = 'black',alpha = 0.1)
            #plt.text(0,np.mean(ylims),'Potentially Incomplete',horizontalalignment = 'right',fontsize=14,verticalalignment = 'top', rotation='vertical')
            plt.xlim(xlims); plt.ylim(ylims);
            plt.xlabel('Log Mass [$\mathrm{M_\odot}$]')
            if multiplicity == 'Properties':
                plt.ylabel('Fraction')
                plt.ylim([0,1])
            elif multiplicity == 'Density Separate':
                plt.ylabel(r'Number Density [$\mathrm{pc}^{-3}$]')
            elif multiplicity == 'Mass Density Separate':
                plt.ylabel(r'Mass Density [$\frac{\mathrm{M_\odot}}{pc^3}$]')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            # if filename is not None:
            #     plt.text(0.7,0.9,filename,transform = plt.gca().transAxes,fontsize=14,horizontalalignment = 'left')
        else:
            return logmasslist,o1,o2,o3
    if multiplicity == 'MF' or multiplicity == 'Density' or multiplicity == 'Mass Density':
        if plot == True:
            if bins == 'continous':
                plt.plot(logmasslist,o1,marker = '^',label = 'Raw Data')
                if only_filter is False:
                    plt.plot(logmasslist_filt,o1_filt,marker = '^',linestyle = ':',label = 'After Corrections')
            elif bins == 'observer':
                if multiplicity == 'MF':
                    if style == 'box':
                        for i in range(len(logmasslist)-1):
                            plt.fill_between([logmasslist[i],logmasslist[i+1]],o1[i]+Psigma(o3[i],o2[i]),o1[i]-Psigma(o3[i],o2[i]),alpha = 0.6,color = '#ff7f0e')
                        if only_filter is False:
                            for i in range(len(logmasslist_filt)-1):
                                plt.fill_between([logmasslist_filt[i],logmasslist_filt[i+1]],o1_filt[i]+Psigma(o3_filt[i],o2_filt[i]),o1_filt[i]-Psigma(o3_filt[i],o2_filt[i]),alpha = 0.3,color = '#1f77b4',hatch=r"\\")
                    elif style == 'line':
                        error = []
                        for o3_err,o2_err in zip(o3,o2):
                            error.append(Psigma(o3_err,o2_err))
                        error = np.array(error,dtype=float);o1 = np.array(o1,dtype=float)
                        plt.fill_between(logmasslist,o1+error,o1-error,alpha = 0.6,color = '#ff7f0e')
                        plt.plot(logmasslist,o1,color = 'black')
                        if only_filter is False:
                            error_filt = []
                            for o3_err,o2_err in zip(o3_filt,o2_filt):
                                error_filt.append(Psigma(o3_err,o2_err))
                            error_filt = np.array(error_filt,dtype=float);o1_filt = np.array(o1_filt,dtype=float)
                            plt.fill_between(logmasslist_filt,o1_filt+error_filt,o1_filt-error_filt,alpha = 0.6,color = '#1f77b4',hatch=r"\\")
                            plt.plot(logmasslist_filt,o1_filt,color = 'black',linestyle = '--')
                else:
                    plt.plot(logmasslist,np.log10(o1),marker = '^')
                    plt.fill_between(logmasslist,np.log10(o1+o2),np.log10(o1)-(np.log10(o1+o2)-np.log10(o1)),alpha = 0.3)
                    if only_filter is False:
                        plt.plot(logmasslist_filt,o1_filt,marker = '^',linestyle = ':',label = 'After Corrections')
            observation_mass_center = [0.0875,0.105,0.1125,0.225,0.45,0.875,1.125,1.175,2,4.5,6.5,12.5]
            observation_mass_width = [0.0075,0.045,0.0375,0.075,0.15,0.125,0.125,0.325,0.4,1.5,1.5,4.5]
            observation_MF = [0.19,0.20,0.19,0.23,0.3,0.42,0.5,0.47,0.68,0.81,0.89,0.93]
            observation_MF_err = [0.07,0.04,0.03,0.02,0.02,0.03,0.04,0.03,0.07,0.06,0.05,0.04]
            plt.xlabel('Log Mass [$\mathrm{M_\odot}$]')
            if multiplicity == 'MF':
                plt.ylabel('Multiplicity Fraction')
                for i in range(len(observation_mass_center)):
                    if i == 0:
                        temp_label = 'Observations'
                    else:
                        temp_label = None
                    plt.errorbar(np.log10(observation_mass_center[i]),observation_MF[i],yerr = observation_MF_err[i],xerr = [[np.log10(observation_mass_center[i])-np.log10(observation_mass_center[i]-observation_mass_width[i])],[np.log10(observation_mass_center[i]+observation_mass_width[i])-np.log10(observation_mass_center[i])]],marker = 'o',capsize = 5,color = 'black',label = temp_label,ls='none')
            elif multiplicity == 'Density':
                plt.ylabel(r'Number Density [$\mathrm{pc}^{-3}$]')
            elif multiplicity == 'Mass Density':
                plt.ylabel(r'Mass Density [$\frac{\mathrm{M_\odot}}{pc^3}$]')
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            if filename is not None:
                plt.text(0.7,0.7,filename,transform = plt.gca().transAxes,fontsize=14,horizontalalignment = 'left')
            handles, labels = plt.gca().get_legend_handles_labels()
            line = mpatches.Patch(label = 'Raw Data',color='#ff7f0e',alpha = 0.6)
            handles.extend([line])
            if only_filter is False:
                line1 = mpatches.Patch(label = 'After Corrections',color='#1f77b4',alpha = 0.3, hatch=r"\\" )
                handles.extend([line1])
            if multiplicity == 'MF':
                plt.legend(handles = handles,fontsize=14)
                xlims = plt.xlim(); ylims = plt.ylim()
                plt.fill_between(np.linspace(xlims[0],0,20),ylims[1],ylims[0],color = 'black',alpha = 0.1)
                plt.xlim(xlims); plt.ylim(ylims);  plt.xlabel('Log Mass [$\mathrm{M_\odot}$]')
        else:
            return logmasslist,o1,o2,o3    
    if multiplicity == 'CF':
        if plot == True:
            if bins == 'continous':
                plt.plot(logmasslist,o1,marker = 'o',label = 'Raw Data')
                if only_filter is False:
                    plt.plot(logmasslist_filt,o1_filt,marker = 'o',label = 'After Corrections',linestyle = ':')
            elif bins == 'observer':
                if style =='box':
                    for i in range(len(logmasslist)-1):
                        plt.fill_between([logmasslist[i],logmasslist[i+1]],o1[i]+Lsigma(o3[i],o2[i]),o1[i]-Lsigma(o3[i],o2[i]),color = '#ff7f0e',alpha = 0.6)
                    if only_filter is False:
                        for i in range(len(logmasslist_filt)-1):
                            plt.fill_between([logmasslist_filt[i],logmasslist_filt[i+1]],o1_filt[i]+Lsigma(o3_filt[i],o2_filt[i]),o1_filt[i]-Lsigma(o3_filt[i],o2_filt[i]),alpha = 0.3,color = '#1f77b4',hatch=r"\\")
                elif style == 'line':
                    error = []
                    for o3_err,o2_err in zip(o3,o2):
                        error.append(Lsigma(o3_err,o2_err))
                    error = np.array(error, dtype=float);o1 = np.array(o1, dtype=float)
                    plt.fill_between(logmasslist,o1+error,o1-error,alpha = 0.6,color = '#ff7f0e')
                    plt.plot(logmasslist,o1,color = 'black')
                    if only_filter is False:
                        error_filt = []
                        for o3_err,o2_err in zip(o3_filt,o2_filt):
                            error_filt.append(Lsigma(o3_err,o2_err))
                        error_filt = np.array(error_filt,dtype=float);o1_filt = np.array(o1_filt,dtype=float)
                        plt.fill_between(logmasslist_filt,o1_filt+error_filt,o1_filt-error_filt,alpha = 0.6,color = '#1f77b4',hatch=r"\\")
                        plt.plot(logmasslist_filt,o1_filt,color = 'black',linestyle = '--')
            observation_mass_center = [0.0875,0.105,0.1125,0.225,0.45,1,1.175,2,4.5,6.5,12.5]
            observation_mass_width = [0.0075,0.045,0.0375,0.075,0.15,0.25,0.325,0.4,1.5,1.5,4.5]
            observation_CF = [0.19,0.20,0.21,0.27,0.38,0.60,0.62,0.99,1.28,1.55,1.8]
            observation_CF_err = [0.07,0.04,0.03,0.03,0.03,0.04,0.04,0.13,0.17,0.24,0.3]
            plt.xlabel('Log Mass [$\mathrm{M_\odot}$]')
            plt.ylabel('Companion Frequency')
            for i in range(len(observation_mass_center)):
                if i == 0:
                    temp_label = 'Observations'
                else:
                    temp_label = None
                plt.errorbar(np.log10(observation_mass_center[i]),observation_CF[i],yerr = observation_CF_err[i],xerr = [[np.log10(observation_mass_center[i])-np.log10(observation_mass_center[i]-observation_mass_width[i])],[np.log10(observation_mass_center[i]+observation_mass_width[i])-np.log10(observation_mass_center[i])]],marker = 'o',capsize = 5,color = 'black',label = temp_label,ls='none')
            if label is not None:
                plt.text(0.7,0.7,label,transform = plt.gca().transAxes,fontsize=14,horizontalalignment = 'left')
            handles, labels = plt.gca().get_legend_handles_labels()
            line = mpatches.Patch(label = 'Raw Data',color='#ff7f0e',alpha = 0.6)
            handles.extend([line])
            if only_filter is False:
                line1 = mpatches.Patch(label = 'After Corrections',color='#1f77b4',alpha = 0.3, hatch=r"\\" )
                handles.extend([line1])
            xlims = plt.xlim(); ylims = plt.ylim()
            plt.fill_between(np.linspace(xlims[0],0,20),ylims[1],ylims[0],color = 'black',alpha = 0.1)
            plt.xlim(xlims); plt.ylim(ylims);  plt.xlabel('Log Mass [$\mathrm{M_\odot}$]')
            plt.legend(handles = handles,fontsize=14)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
            plt.figure(figsize=(6,6))
        else:
            return logmasslist,o1,o2,o3

def Time_Evolution_Plots(which_plot,Master_File,file,steps = 1,target_mass = 1,T = None,dt = None,target_age = 0.5,filename = None,min_age = 0,read_in_result = True,start = 0,upper_limit = 1.3,lower_limit = 0.7,plot = True,multiplicity = 'MF',zero = 'Consistent Mass',select_by_time = True,rolling_avg = False,rolling_window = 0.1,time_norm = 'tff',min_time_bin = 0.2,adaptive_binning = True,adaptive_no = 20,x_axis = 'mass density',description = None,label=None):
    '''
    Create a plot for a property that evolves through the simulation.

    Inputs
    ----------
    which_plot: string
    The plot to be made.

    Master_File: list of lists of star system objects
    The entire simulation with system assignment. Only required for time evolution plots and filter on.

    file:list of sinkdata objects
    The original sinkdata file.

    Parameters
    ----------
    steps: int,optional
    The number of snapshots per bin in multiplicity over time.

    target_mass: int,float,optional
    The mass of the primaries of the systems of interest.

    T: int,float,optional
    The time from end of the simulation to select from

    dt: int,float,optional
    The tolerance of T. For example, if the total runtime is 10 Myr, T = 2 and dt = 0.2, then it looks at stars formed between 7.9-8.1 Myrs.

    target_age:int,float,optional
    The maximum age for the YSO multiplicities.

    filename:string
    The name of the original file. It will be labelled on the plot if provided.

    min_age:int,float,optional
    The minimum age for the YSO multiplicity

    read_in_result: bool,optional
    Whether to perform system assignment again or just read it in.

    start: int,optional
    Starting point of multiplicity time evolution

    upper_limit: int,float,optional
    The upper mass limit of the primaries of systems of interest.

    lower_limit: int,float,optional
    The lower mass limit of the primaries of systems of interest.

    plot: bool,optional
    Whether to plot the data or just return it.

    multiplicity: bool,optional
    Whether to plot for the multiplicity properties(only by mass plot), multiplicity fraction or Companion Frequency.

    zero: string,optional
    Whether to set the zero age as 'formation' (where the star formed) or 'consistent mass' (where the star stopped accreting)

    select_by_time: bool,optional
    Whether to look at average multiplicity for all stars or only those in a window.

    rolling_avg: bool,optional
    Whether to use a rolling average or not.

    rolling_window: int,float,optional
    Time to include in the rolling average. [in Myr]
    
    time_norm : str,optional
    Whether to use the simulation time in Myr('Myr'), in free fall time('tff'), or in free fall time and sqrt alpha ('atff')
    
    min_time_bin: int,optional
    The minimum time bin to plot on the time histogram
    
    adaptive_binning: bool,optional
    Whether to adopt adaptive binning (same no of stars in each bin)
    
    adaptive_no: int,optional
    The number of stars in each bin
    
    x_axis: string,optional
    Whether to plot the MF/CF with the formation time/density/mass density
    
    description: string,optional
    What to save the name of the Multiplicity Tracker plot under.

    Returns
    -------
    x_vals: list
    The bins.

    weights:list
    The weights of each bin

    Examples
    -------
    1) Time_Evolution_Plots("Multiplicity Time Evolution",M2e4_C_M_J_2e7_systems,M2e4_C_M_J_2e7,multiplicity = 'Fraction',target_mass = 1')
    The multiplicity at every time for the given target mass.

    2)Time_Evolution_Plots("Multiplicity Lifetime Evolution",M2e4_C_M_J_2e7_systems,M2e4_C_M_J_2e7,multiplicity = 'Fraction',target_mass = 1,T = [1,2,3],dt = [0.5,0.5,0.5]')
    The multiplicity of stars of the target mass born at the given times.

    3)Time_Evolution_Plots("YSO Multiplicity",M2e4_C_M_J_2e7_systems,M2e4_C_M_J_2e7,min_age = 0,target_age = 0.5)
    The multiplicity of stars of younger than the target age and older than the minimum age.
    '''
    if label is None: label = path.basename(filename)
    if which_plot == 'Multiplicity Time Evolution':
        if Master_File is None:
            print('provide master file')
            return
        elif filename is None:
            print('Provide the filename')
        MFCF_Time_Evolution(file,Master_File,filename,steps = steps,target_mass=target_mass,read_in_result=read_in_result,start = start,upper_limit=upper_limit,lower_limit=lower_limit,plot = plot,time_norm = time_norm,multiplicity=multiplicity,rolling_avg=rolling_avg,rolling_window_Myr=rolling_window)
    print('\n'+which_plot+'\n')
    if which_plot == 'Multiplicity Lifetime Evolution':
        if Master_File is None:
            print('provide master file')
            return
        if plot is False:
            print('Use Plot == True')
            return
        multiplicity_and_age_combined(file,Master_File,filename = filename,T_list=T,dt_list=dt,upper_limit=upper_limit,lower_limit=lower_limit,target_mass=target_mass,zero = zero,multiplicity=multiplicity,rolling_avg=rolling_avg,rolling_window_Myr=rolling_window,min_time_bin=min_time_bin,adaptive_binning=adaptive_binning,adaptive_no=adaptive_no,description=description,label=label)
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
    if which_plot == 'Multiplicity vs Formation':
        plt.figure(figsize = (6,6))
        multiplicity_vs_formation(file,Master_File,upper_limit=upper_limit,lower_limit=lower_limit,target_mass=target_mass,zero=zero,multiplicity=multiplicity,min_time_bin=min_time_bin,adaptive_no=adaptive_no,x_axis=x_axis,plot = True,label=label)
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        #plt.savefig(multiplicity+'_vs_formation_'+label+'.png',dpi = 150, bbox_inches='tight')
    if which_plot == 'YSO Multiplicity':
        plt.figure(figsize = (6,6))
        if Master_File is None:
            print('provide master file')
            return
        if filename is None:
            print('Please Provide filename')
        mul1,mul1_err,cou1,av1, pcount = YSO_multiplicity(file,Master_File,target_age = target_age,min_age=min_age, return_all=True)
        times = []
        prop_times = []
        start_snap = Mass_Creation_Finder(file,min_mass = 0)
        start_time = file[start_snap].t*code_time_to_Myr
        for i in range(len(file)):
            times.append(file[i].t*code_time_to_Myr - start_time)
        for i in range(len(file)):
            prop_times.append(file[i].t)
        #end_snap = closest(prop_times,prop_times[-1]-target_age,param = 'index')

        rolling_window = time_to_snaps(rolling_window,file)
        if rolling_window%2 == 0:
            rolling_window -= 1
        rolling_window = int(rolling_window)
        
        if time_norm != 'Myr':
            prop_times = np.array(prop_times)
            ff_t = t_ff(file_properties(filename,param = 'm'),file_properties(filename,param = 'r'))
            if time_norm == 'atff':
                alpha = file_properties(filename,param = 'alpha')
                ff_t = ff_t*np.sqrt(alpha)
            prop_times = (prop_times/ff_t)
        else:
            prop_times = np.array(prop_times)*code_time_to_Myr

        
        if rolling_avg is True:
            #prop_times = rolling_average(prop_times,rolling_window)
            mul1_err = rolling_average(mul1+mul1_err,rolling_window)- rolling_average(mul1,rolling_window)
            mul1 = rolling_average(mul1,rolling_window)
            cou1 = rolling_average(cou1,rolling_window)
            av1 = rolling_average(av1,rolling_window)
            pcount = rolling_average(pcount,rolling_window)
        
        # if rolling_avg is True:
        #     # print("Overwriting rolling average window to 0.5 Myr for single YSO plot")
        #     # rolling_window=0.5
        #     time_edges = np.linspace(np.min(prop_times),np.max(prop_times),num=int(len(prop_times)/time_to_snaps(rolling_window,file)+1))
        #     time_bins = (time_edges[1:]+time_edges[:-1])/2
        #     tot_mass = av1*cou1
        #     all_in_bin,_,_ = stats.binned_statistic(prop_times,cou1,statistic='sum',bins = time_edges)
        #     primary_in_bin,_,_ = stats.binned_statistic(prop_times,pcount,statistic='sum',bins = time_edges)
        #     mass_in_bin,_,_ = stats.binned_statistic(prop_times,tot_mass,statistic='sum',bins = time_edges)
        #     #replace values
        #     mul1 = primary_in_bin/all_in_bin
        #     mul1_err = np.array([ Psigma(all_in_bin[i],primary_in_bin[i]) for i in range(len(primary_in_bin)) ])
        #     print(mul1)
        #     print(mul1_err)
        #     print(all_in_bin)
        #     prop_times = time_bins
        #     cou1 = all_in_bin
        #     av1=mass_in_bin/all_in_bin
            
            
        
        plt.plot(prop_times,mul1,label = '< '+str(target_age)+' Myr stars', color='b')
        plt.fill_between(prop_times, np.clip(mul1-mul1_err,0,1),y2=np.clip(mul1+mul1_err,0,1),alpha=0.3)
        
        left_limit = plt.xlim()[0]
        right_limit = plt.xlim()[1]
        
        plt.fill_betweenx(np.linspace(0.35,0.5,100),left_limit,right_limit,color = 'orange',alpha = 0.3)
        plt.fill_betweenx(np.linspace(0.3,0.4,100),left_limit,right_limit,color = 'black',alpha = 0.3)
        plt.fill_betweenx(np.linspace(0.25,0.15,100),left_limit,right_limit,color = 'purple',alpha = 0.3)
        textshift = 0.2*np.ptp(plt.xlim())
        plt.text(0.1+left_limit+textshift,0.45,'Class 0 Perseus',fontsize = 16,horizontalalignment = 'left',zorder=10)
        plt.text(0.1+left_limit+textshift,0.32,'Class 0 Orion',fontsize = 16,horizontalalignment = 'left',zorder=10)
        plt.text(0.1+left_limit+textshift,0.2,'Class I Orion',fontsize = 16,horizontalalignment = 'left',zorder=10)
        plt.ylim([0,np.min([1.0,plt.ylim()[1]+0.05])])
        
        if time_norm == 'Myr':
            plt.xlabel('Time [Myr]')
        elif time_norm == 'tff':
            plt.xlabel(r'Time [$t_\mathrm{ff}$]')
        elif time_norm == 'atff':
            plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
        plt.ylabel('YSO Multiplicity Fraction', color='b')

        plt.xlim((left_limit,right_limit))
        ax1 = plt.gca()
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Log Number of YSOs', color='r')
        ax2.plot(prop_times, np.log10(cou1), color='r',zorder=0)
        ax2.tick_params(axis='y', labelcolor='r')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.savefig('YSO_MF_'+label+'.png',dpi = 150, bbox_inches='tight')
        
        plt.figure(figsize = (6,6))
        # if label is not None:
        #     plt.text(0.7,0.7,label,transform = plt.gca().transAxes,fontsize = 12,horizontalalignment = 'left')
        plt.plot(prop_times,np.log10(cou1),label = '< '+str(target_age)+' Myr stars')
        #plt.legend(fontsize=14)
        if time_norm == 'Myr':
            plt.xlabel('Time [Myr]')
        elif time_norm == 'tff':
            plt.xlabel(r'Time [$t_\mathrm{ff}$]')
        elif time_norm == 'atff':
            plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
        plt.ylabel('Log Number of YSOs')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        #plt.legend(fontsize=14)
        plt.savefig('YSO_count_'+label+'.png',dpi = 150, bbox_inches='tight')
        
        plt.figure(figsize = (6,6))
        plt.plot(prop_times,np.log10(av1),label = '< '+str(target_age)+' Myr stars')
        #[start_snap:end_snap]
        #plt.yscale('log')
        #plt.plot(times,av2,label = 'Consistent Mass')
        if time_norm == 'Myr':
            plt.xlabel('Time [Myr]')
        elif time_norm == 'tff':
            plt.xlabel(r'Time [$t_\mathrm{ff}$]')
        elif time_norm == 'atff':
            plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
        plt.ylabel(r'Log Average YSO Mass [$\mathrm{M_\odot}$]')
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.savefig('YSO_avg_mass_'+label+'.png',dpi = 150, bbox_inches='tight')

#Function that contains all the plots
def Plots(which_plot,Master_File,file,filename = None,systems = None,snapshot= -1,target_mass=1,target_age=1,upper_limit = 1.3,lower_limit = 0.7,mass_break = 2,T = [1],dt = [0.5],min_age = 0,all_companions = True,bins = None,log = True,compare = False,plot = True,multiplicity = 'MF',steps = 1,read_in_result = True,start = 0,zero = 'Formation',select_by_time = True,filters = ['q_filter','time_filter'],avg_filter_snaps_no = 10,q_filt_min = 0.1,time_filt_min = 0.1,only_filter = True,rolling_avg = True,rolling_window_Myr = 0.1,time_norm = 'tff',min_time_bin = 0.2,adaptive_binning = True,adaptive_no = 20,x_axis = 'mass density',description = None, label=None,filter_in_class = True,MFCF_plot_style = 'line',plot_intermediate_filters = False): 
    '''
    Create a plot or gives you the values to create a plot for the whole system.

    Inputs
    ----------
    which_plot: string
    The plot to be made.
    
    Master_File: list of lists of star system objects
    The entire simulation with system assignment. Only required for time evolution plots and filter on.

    file:list of sinkdata objects
    The original sinkdata file.

    Parameters
    ----------
    
    filename:string
    The name of the original file. It will be labelled on the plot if provided.
    
    systems: list of star system objects
    The systems you want to analyze (1 snapshot or a filtered snapshot).
    
    snapshot: int,float
    The snapshot number you want to look at. Only required for IMF comparisons and filter on.

    target_mass: int,float,optional
    The mass of the primaries of the systems of interest.

    target_age:int,float,optional
    The maximum age for the YSO multiplicities.

    upper_limit: int,float,optional
    The upper mass limit of the primaries of systems of interest.

    lower_limit: int,float,optional
    The lower mass limit of the primaries of systems of interest.

    mass_break: int,float,optional
    The spacing between masses in log space (important for multiplicity fraction)

    T: list,optional
    The times from end of the simulation to select from. 

    dt: list,optional
    The tolerance of T. 

    min_age:int,float,optional
    The minimum age for the YSO multiplicity

    all_companions: bool,optional
    Whether to include all companions in the mass ratio or semi major axes.

    bins: int,float,list,array,string,optional
    The bins for the histograms.

    log: bool,optional
    Whether to plot the y data on a log scale.

    compare: bool,optional
    Whether to include the IMF for comparison.

    plot: bool,optional
    Whether to plot the data or just return it.

    multiplicity: bool,optional
    Whether to plot for the multiplicity properties(only by mass plot), multiplicity fraction or Companion Frequency.

    steps: int,optional
    The number of snapshots per bin in multiplicity over time.

    read_in_result: bool,optional
    Whether to perform system assignment again or just read it in.

    start: int,optional
    Starting point of multiplicity time evolution

    zero: string,optional
    Whether to set the zero age as 'formation' (where the star formed) or 'consistent mass' (where the star stopped accreting)

    select_by_time: bool,optional
    Whether to look at average multiplicity for all stars or only those in a window.

    filters: list of strings,None,optional
    Whether to use a filter or not. The choices are 'time_filter','q_filter' or 'average_filter'
    
    avg_filter_snaps_no: int,optional
    The number of snapshots to average over with the average filter
    
    q_filt_min: float,optional
    The minimum q to use in the q filter
    
    time_filt_min: float,optional
    The minimum time that companions should have been in a system
     
    only_filter: bool,optional
    Whether to only look at the filter (True) or to plot both the filter and unfiltered data (False)..

    rolling_avg: bool,optional
    Whether to use a rolling average or not.

    rolling_window: int,float,optional
    The time in the rolling window.[in Myr]
    
    time_norm : str,optional
    Whether to use the simulation time in Myr('Myr'), in free fall time('tff'), or in free fall time and sqrt alpha ('atff')
    
    min_time_bin: int,optional
    The minimum time bin to plot on the time histogram
    
    adaptive_binning: bool,optional
    Whether to adopt adaptive binning (same no of stars in each bin)
    
    adaptive_no: int,optional
    The number of stars in each bin
    
    x_axis: string,optional
    Whether to plot the MF/CF with the formation time/density/mass density
    
    description: string,optional
    What to save the name of the Multiplicity Tracker plot under.
    
    filter_in_class: bool,optional
    If the filter is saved in the class definition.
    
    MFCF_plot_style: string,optional
    Whether to use a line or box style for the MF and CF plots

    Returns
    -------
    x_vals: list
    The bins.

    weights:list
    The weights of each bin
    '''
    One_System_Plots = ['System Mass','Primary Mass','Semi Major Axis','Angle','Mass Ratio','Semi Major Axis vs q', 'Eccentricity']
    Time_Evo_Plots = ['Multiplicity Time Evolution','Multiplicity Lifetime Evolution','Multiplicity vs Formation','YSO Multiplicity']
    if label is None: label=filename
    if which_plot in One_System_Plots:
        if plot == True:
            One_Snap_Plots(which_plot,Master_File = Master_File,file = file,systems = systems,filename = filename,snapshot = snapshot,upper_limit = upper_limit,lower_limit = lower_limit,target_mass = target_mass,all_companions = all_companions,bins = bins,log = log,compare = compare,plot = plot,read_in_result = read_in_result,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter = only_filter,label=label,filter_in_class = filter_in_class,plot_intermediate_filters = plot_intermediate_filters)
        else:
            return One_Snap_Plots(which_plot,Master_File = Master_File,file = file,systems = systems,filename = filename,snapshot = snapshot,upper_limit = upper_limit,lower_limit = lower_limit,target_mass = target_mass,all_companions = all_companions,bins = bins,log = log,compare = compare,plot = plot,read_in_result = read_in_result,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter = only_filter,label=label,filter_in_class = filter_in_class,plot_intermediate_filters = plot_intermediate_filters)
    elif which_plot == 'Multiplicity':
        if plot == True:
            Multiplicity_One_Snap_Plots(Master_File,file,systems = systems,snapshot = snapshot,filename = filename,plot = plot,multiplicity = multiplicity,mass_break=mass_break,bins = bins,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter = only_filter,label=label,filter_in_class = filter_in_class,style = MFCF_plot_style)
        else:
            return Multiplicity_One_Snap_Plots(Master_File,file,systems = systems,snapshot = snapshot,filename = filename,plot = plot,multiplicity = multiplicity,mass_break=mass_break,bins = bins,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter = only_filter,label=label,filter_in_class = filter_in_class,style = MFCF_plot_style)
    elif which_plot in Time_Evo_Plots:
        if plot == True:
            Time_Evolution_Plots(which_plot,Master_File,file,filename=filename,steps = steps,target_mass = target_mass,T = T,dt = dt,target_age = target_age,min_age = min_age,read_in_result = read_in_result,start = start,upper_limit = upper_limit,lower_limit = lower_limit,plot = plot,multiplicity = multiplicity,zero = zero,select_by_time = select_by_time,rolling_avg=rolling_avg,rolling_window=rolling_window_Myr,time_norm = time_norm,min_time_bin = min_time_bin,adaptive_binning = adaptive_binning,adaptive_no = adaptive_no,x_axis = x_axis,description=description)
        else:
            return Time_Evolution_Plots(which_plot,Master_File,file,filename=filename,steps = steps,target_mass = target_mass,T = T,dt = dt,target_age = target_age,min_age = min_age,read_in_result = read_in_result,start = start,upper_limit = upper_limit,lower_limit = lower_limit,plot = plot,multiplicity = multiplicity,zero = zero,select_by_time = select_by_time,rolling_avg=rolling_avg,rolling_window=rolling_window_Myr,time_norm = time_norm,min_time_bin = min_time_bin,adaptive_binning = adaptive_binning,adaptive_no = adaptive_no,x_axis = x_axis,description=description)

def Multiplicity_One_Snap_Plots_Filters(Master_File,file,systems = None,snapshot = -1,filename = None,plot = True,multiplicity = 'MF',mass_break=2,bins = 'observer',filters = ['q_filter','time_filter'],avg_filter_snaps_no = 10,q_filt_min = 0.1,time_filt_min = 0.1,filter_display = 'sub-filter',label=None,filter_in_class = True,include_error = True):
    if filter_display != 'sub-filter':
        if filter_display == 'none':
            filters = [None]
            only_filter = True
        if filter_display == 'top-level':
            only_filter = True
        if filter_display == 'comparison':
            only_filter = False
        Multiplicity_One_Snap_Plots(Master_File,file,systems = systems,snapshot = snapshot,filename = filename,plot = plot,multiplicity = multiplicity,mass_break=mass_break,bins = bins,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter = only_filter,label=label,filter_in_class = filter_in_class)
    else:
        filter_labels = ['Simulation',r'q>%3.2g'%(q_filt_min),r'$t_\mathrm{comp}$>%3.2g Myr'%(time_filt_min)]
        #filter_labels.append(filter_labels[-2]+' & '+filter_labels[-1])
        filter_labels.append('All corrections')
        filter_names = [[None],['q_filter'],['time_filter'],['q_filter','time_filter']]
        #linestyles = ['-',':','--','-.']
        linestyles = ['-','-','-','-','-','-']
        colorlist, _ = set_colors_and_styles(None, None, len(filter_names), dark=True, sequential=True)
        for i in range(len(filter_names)):
            logmasslist,o1,o2,o3 = Multiplicity_One_Snap_Plots(Master_File,file,systems = systems,snapshot = snapshot,filename = filename,plot = False,multiplicity = multiplicity,mass_break=mass_break,bins = bins,filters = filter_names[i],avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter = True,label=label,filter_in_class = filter_in_class)
            error = []
            for o3_err,o2_err in zip(o3,o2):
                if multiplicity == 'MF':
                    error.append(Psigma(o3_err,o2_err))
                else:
                    error.append(Lsigma(o3_err,o2_err))
            error = np.array(error, dtype=float);o1 = np.array(o1, dtype=float)
            if include_error is True:
                if filter_labels[i] == 'Simulation' or filter_labels[i] == filter_labels[-1]:
                    plt.fill_between(logmasslist,o1+error,o1-error,alpha = 0.2, color=colorlist[i])
            plt.plot(logmasslist,o1,linestyle = linestyles[i],label = filter_labels[i], color=colorlist[i])
            plt.xlabel('Log Mass [$\mathrm{M_\odot}$]')
        if multiplicity == 'MF':
            observation_mass_center = [0.0875,0.205,0.1125,0.225,0.45,1,0.875,1.125,1.175,2,4.5,6.5,12.5]
            observation_mass_width = [0.0075,0.045,0.0375,0.075,0.15,0.25,0.125,0.125,0.325,0.4,1.5,1.5,4.5]
            observation_MF = [0.19,0.20,0.19,0.23,0.3,np.nan,0.42,0.5,0.47,0.68,0.81,0.89,0.93]
            observation_MF_err = [0.07,0.04,0.03,0.02,0.02,np.nan,0.03,0.04,0.03,0.07,0.06,0.05,0.04]
            for i in range(len(observation_mass_center)):
                if i == 0:
                    temp_label = 'Observations'
                else:
                    temp_label = None
                plt.errorbar(np.log10(observation_mass_center[i]),observation_MF[i],yerr = observation_MF_err[i],xerr = [[np.log10(observation_mass_center[i])-np.log10(observation_mass_center[i]-observation_mass_width[i])],[np.log10(observation_mass_center[i]+observation_mass_width[i])-np.log10(observation_mass_center[i])]],marker = 'o',capsize = 5,color = 'black',label = temp_label,ls='none')
            plt.ylabel('Multiplicity Fraction')
            bottom,top = plt.ylim()
            if top >1:
                plt.ylim(bottom,1)
        if multiplicity == 'CF':
            observation_mass_center = [0.0875,0.205,0.1125,0.225,0.45,1,0.875,1.125,1.175,2,4.5,6.5,12.5]
            observation_mass_width = [0.0075,0.045,0.0375,0.075,0.15,0.25,0.125,0.125,0.325,0.4,1.5,1.5,4.5]
            observation_CF = [0.19,0.20,0.21,0.27,0.38,0.60,np.nan,np.nan,0.62,0.99,1.28,1.55,1.8]
            observation_CF_err = [0.07,0.04,0.03,0.03,0.03,0.04,np.nan,np.nan,0.04,0.13,0.17,0.24,0.3]
            for i in range(len(observation_mass_center)):
                if i == 0:
                    temp_label = 'Observations'
                else:
                    temp_label = None
                plt.errorbar(np.log10(observation_mass_center[i]),observation_CF[i],yerr = observation_CF_err[i],xerr = [[np.log10(observation_mass_center[i])-np.log10(observation_mass_center[i]-observation_mass_width[i])],[np.log10(observation_mass_center[i]+observation_mass_width[i])-np.log10(observation_mass_center[i])]],marker = 'o',capsize = 5,color = 'black',label = temp_label,ls='none')
            plt.ylabel('Companion Frequency')
            bottom,top = plt.ylim()
            if top >3:
                plt.ylim(bottom,3)
        xlims = plt.xlim(); ylims = plt.ylim()
        plt.fill_between(np.linspace(xlims[0],0,20),ylims[1],ylims[0],color = 'black',alpha = 0.1)
        plt.xlim(xlims); plt.ylim(ylims)
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14)
        plt.legend(fontsize = 14,labelspacing=0)
        
def Multi_Plot(which_plot,Systems,Files,Filenames,Snapshots = None,bins = None,log = False,upper_limit = 1.3,lower_limit = 0.7,target_mass = 1,target_age = 0.5,min_age = 0,multiplicity = 'MF',steps = 1,read_in_result = True,all_companions = True,start = 0,select_by_time = True,filters = ['q_filter','time_filter'],avg_filter_snaps_no = 10,q_filt_min = 0.1,time_filt_min = 0.1,normalized = True,norm_no = 100,time_plot = 'consistent mass',rolling_avg=True,rolling_window=0.1,time_norm = 'tff',adaptive_no = [20],adaptive_binning = True,x_axis = 'mass density',zero = 'Formation',description = None,labels=None,filter_in_class = True, colors=None):
    '''
    Creates distribution plots for more than one file
    Inputs
    ----------
    which_plot: string
    The plot to be made.

    Systems: list of list of star system objects
    All of the Systems from all of the files you want to see.

    Files:list of list of sinkdata objects
    The list of all the files you want to see.

    Filenames: list of strings
    The names of the files that you want to see.

    Snapshots: int,float
    The snapshot number you want to look at. By default, it looks at the last one.

    Parameters
    ----------
    log: bool,optional
    Whether to plot the y data on a log scale.

    upper_limit: int,float,optional
    The upper mass limit of the primaries of systems of interest.

    lower_limit: int,float,optional
    The lower mass limit of the primaries of systems of interest.

    target_mass: int,float,optional
    The mass of the primaries of the systems of interest.

    multiplicity: bool,optional
    Whether to plot for the multiplicity properties(only by mass plot), multiplicity fraction or Companion Frequency.

    all_companions: bool,optional
    Whether to include all companions in the mass ratio or semi major axes.

    filtered: bool,optional:
    Whether to include the filtered results or the unfiltered results

    normalized:bool,optional:
    Whether to normalize the systems to a certain number

    norm_no: int,optional:
    The number of systems to normalize to.

    time_plot:str,optional:
    Whether to plot the consistent mass or all of the stars in the multiplicity time evolution.
    
    rolling_avg: bool,optional:
    Whether to use a rolling average or not.
    
    rolling_window: int,optional:
    How many points to include in the rolling average window.[in Myr]
    
    time_norm : str,optional
    Whether to use the simulation time in Myr('Myr'), in free fall time('tff'), or in free fall time and sqrt alpha ('atff')
    
    adaptive_no: list,optional
    The number of stars in each bin for each file
    
    adaptive_binning: bool,optional
    Whether to adopt adaptive binning (same no of stars in each bin)
    
    x_axis: string,optional
    Whether to plot the MF/CF with the formation time/density/mass density
    
    zero: string,optional
    Whether to set the zero age as 'formation' (where the star formed) or 'consistent mass' (where the star stopped accreting)

    description: string,optional
    What to save the name of the YSO plot under.
    
    filter_in_class: bool,optional
    If the filter is saved in the class definition.
    
    Examples
    ----------
    1) Multi_Plot('Mass Ratio',Systems,Files,Filenames,normalized=True)
    '''  
    
    if colors is None:
        colors, _ = set_colors_and_styles(None, None, len(Filenames),  dark=True)
    
    adjust_ticks=True
    if labels is None: labels=Filenames
    if which_plot == 'Multiplicity vs Formation':
        multiplicity_vs_formation_multi(Files,Systems,Filenames,adaptive_no = adaptive_no,T_list = None,dt_list = None,upper_limit=upper_limit,lower_limit = lower_limit,target_mass = target_mass,zero = zero,multiplicity = multiplicity,adaptive_binning = adaptive_binning,x_axis = x_axis,labels=labels, colors=colors)
    else:
        if Snapshots == None:
            Snapshots = [[-1]]*len(Filenames)
        Snapshots = list(flatten(Snapshots))
        x = []
        y = []
        if which_plot == 'System Mass' or which_plot == 'Primary Mass' or which_plot == 'Semi Major Axis':
            array = []
            for i in range(len(Files)):
                array.append(primary_total_ratio_axis(Systems[i][Snapshots[i]],lower_limit = lower_limit,upper_limit = upper_limit,all_companions=all_companions,attribute = which_plot,file = Files[i]))
            array = list(flatten(array))
            array = np.log10(array)
            if which_plot == 'Semi Major Axis':
                array = np.array(array)-np.log10(m_to_AU)
            #array = list(array)
            floor = np.floor(np.min(array[np.isfinite(array)]))
            ceiling = np.ceil(np.max(array[np.isfinite(array)]))
        if normalized is True:
            normal_str = 'Normalized '
        else:
            normal_str = ''
        if which_plot == 'System Mass':
            if bins is None:
                bins = np.linspace(floor,ceiling,int(ceiling-floor+1))
            plt.xlabel('Log System Mass [$\mathrm{M_\odot}$]')
            plt.ylabel(normal_str+'Number of systems')
        if which_plot == 'Primary Mass':
            if bins is None:
                bins = np.linspace(floor,ceiling,int(ceiling-floor+1))
            plt.xlabel('Log Primary Mass [$\mathrm{M_\odot}$]')
            plt.ylabel(normal_str+'Number of systems')
        if which_plot == 'Semi Major Axis':
            if bins is None:
                bins = np.linspace(floor,ceiling,int((ceiling-floor)*3/2+1))
        if which_plot == 'Angle':
            if bins is None:
                bins = np.linspace(0,180,10)
            plt.xlabel('Misalignment Angle [°]')
            plt.ylabel(normal_str+'Number of systems')
        if which_plot == 'Mass Ratio':
            if bins is None:
                bins = np.linspace(0,1,11)
            plt.xlabel('q (Companion Mass Ratio)')
            plt.ylabel(normal_str+'Number of systems')
            if all_companions is True:
                plt.ylabel('Number of Companions')
        if which_plot == 'Multiplicity':
            if bins is None:
                bins = 'observer'
            plt.xlabel('Log Mass [$\mathrm{M_\odot}$]')
            if multiplicity == 'MF':
                plt.ylabel('Multiplicity Fraction')
            if multiplicity == 'CF':
                plt.ylabel('Companion Frequency')
        if which_plot == 'Multiplicity':
            error = []
        times = []
        fractions = []
        fractions_err = []
        cons_fracs = []
        cons_fracs_err = []
        nos = []
        avg_mass = []
        og_rolling_window = copy.copy(rolling_window)
        offsets = [0]
        for i in range(1,len(Filenames)):
            inc = 0.2 if labels[i]=='' else 1
            offsets += [offsets[-1]+inc]
        offsets = np.array(offsets)
        if which_plot == 'Mass Ratio':
            offsets = 0.01*offsets
        elif which_plot == 'Angle':
            offsets = 2*offsets
        else:
            offsets = 0.1*offsets  
        for i in tqdm(range(0,len(Filenames)),desc = 'Getting Data',position=0):
            if which_plot == 'Multiplicity':
                a,b,c,d = Plots(which_plot,Systems[i],Files[i],Filenames[i],Systems[i][Snapshots[i]],log = False,plot = False,bins = bins,upper_limit = upper_limit,lower_limit = lower_limit,multiplicity = multiplicity,all_companions = all_companions,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter=True,snapshot = Snapshots[i],filter_in_class = filter_in_class)
                comp_mul_no = c
                sys_no = d
                error_one = []
                for i in range(len(sys_no)):
                    if multiplicity == 'MF':
                        error_one.append(Psigma(sys_no[i],comp_mul_no[i]))
                    elif multiplicity == 'CF':
                        error_one.append(Lsigma(sys_no[i],comp_mul_no[i]))
                error.append(error_one)
                x.append(a)
                y.append(b)
            elif which_plot == 'Multiplicity Time Evolution':
                time,fraction,fraction_err,cons_frac,cons_frac_err = MFCF_Time_Evolution(Files[i],Systems[i],Filenames[i],steps = steps,target_mass=target_mass,read_in_result=read_in_result,start = start,upper_limit=upper_limit,lower_limit=lower_limit,plot = False,time_norm = time_norm,multiplicity=multiplicity)
                if rolling_avg is True:
                    rolling_window = int(time_to_snaps(og_rolling_window,Files[i]))
                    if rolling_window%2 == 0:
                        rolling_window -= 1
                    #time = rolling_average(time,rolling_window)
                    fraction_err = rolling_average(fraction+fraction_err,rolling_window) - rolling_average(fraction,rolling_window)
                    fraction = rolling_average(fraction,rolling_window)
                    cons_frac_err =  rolling_average(cons_frac+cons_frac_err,rolling_window)- rolling_average(cons_frac,rolling_window)
                    cons_frac = rolling_average(cons_frac,rolling_window)
                times.append(time)
                fractions.append(fraction)
                fractions_err.append(fraction_err)
                cons_fracs.append(cons_frac)
                cons_fracs_err.append(cons_frac_err)
            elif which_plot == 'YSO Multiplicity':
                time = []
                for j in range(len(Files[i])):
                    time.append(Files[i][j].t)

                time = np.array(time)
                ff_t = t_ff(file_properties(Filenames[i],param = 'm'),file_properties(Filenames[i],param = 'r'))
                if time_norm == 'atff':
                    time = (time/(ff_t*np.sqrt(file_properties(Filenames[i],param = 'alpha'))))
                elif time_norm == 'tff':
                    time = (time/(ff_t))

                if rolling_avg is True:
                    rolling_window = time_to_snaps(og_rolling_window,Files[i])
                    if rolling_window%2 == 0:
                        rolling_window -= 1
                    rolling_window = int(rolling_window)
                    #time = rolling_average(time,rolling_window)
                times.append(time)
                fraction,fraction_err,no,am = YSO_multiplicity(Files[i],Systems[i],min_age = min_age,target_age = target_age)
                if rolling_avg is True:
                    fraction_err = rolling_average(fraction+fraction_err,rolling_window) - rolling_average(fraction,rolling_window)
                    fraction = rolling_average(fraction,rolling_window)
                    no = rolling_average(no,rolling_window)
                    am = rolling_average(am,rolling_window)
                fractions.append(fraction)
                fractions_err.append(fraction_err)
                nos.append(no)
                avg_mass.append(am)
            else:
                a,b = Plots(which_plot,Systems[i],Files[i],Filenames[i],Systems[i][Snapshots[i]],log = False,plot = False,bins = bins,upper_limit = upper_limit,lower_limit = lower_limit,multiplicity = multiplicity,all_companions = all_companions,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter=True,snapshot = Snapshots[i],filter_in_class = filter_in_class)
                if normalized == True:
                    b = b*norm_no/sum(b)
                x.append(a)
                y.append(b)
        if which_plot == 'Semi Major Axis':
            fig = plt.figure(figsize = (6,6))
            ax1 = fig.add_subplot(111)
            for i in range(len(Files)):
                ax1.step(x[i]-offsets[i],y[i]-offsets[i],label = labels[i], color=colors[i])
            ax1.vlines(np.log10(20),plt.ylim()[0],plt.ylim()[1],colors='k',linestyles='dashed')
            plt.text(np.log10(20)+0.1, 1, r'Grav. softening length', horizontalalignment='left',\
                     verticalalignment='center', fontsize=11)#,rotation='vertical')
            pands = []
            for i in Systems[0][-1]:
                if i.no>1 and lower_limit<=i.primary<=upper_limit:
                    pands.append(i.primary+i.secondary)
            average_pands = np.average(pands)*1.9891e30 
            ax1.set_xlabel('Log Semi Major Axis [AU]',fontsize=14)
            ax1.set_ylabel(normal_str+'Number of systems',fontsize=14)
            if all_companions == True:
                ax1.set_ylabel(normal_str+'Number of Sub-Systems',fontsize=14)
            ax2 = ax1.twiny(); adjust_ticks=False
            ax2.set_xlabel('Log Period [Days]',fontsize=14)
            logperiod_lims = np.log10(2*np.pi*np.sqrt(((10**np.array(ax1.get_xlim())*m_to_AU)**3)/(6.67e-11*average_pands))/(s_to_day))
            ax2.set_xlim(logperiod_lims)
            if (upper_limit == 1.3 and lower_limit == 0.7) or (lower_limit>=5.0):
                # periods = np.linspace(3.5,7.5,num = 5)
                # k = ((10**periods)*24*60*60)
                # smaxes3 = ((6.67e-11*(k**2)*average_pands)/(4*np.pi**2))
                # smaxes = np.log10((smaxes3**(1/3))/m_to_AU)
                # error_values_small = np.array([6,7,9,9,10])
                # error_values_big = np.array([18,27,31,23,21])
                # error_values_comb = (error_values_small+error_values_big)
                # dy_comb = np.sqrt(error_values_comb)
                # ax1.errorbar(smaxes,np.array(error_values_comb)*max(y[0])/max(error_values_comb),yerr=dy_comb*max(y[0])/max(error_values_comb),xerr = (2/3)*0.5*np.ones_like(len(smaxes)),marker = 'o',capsize = 5,color = 'black',label = 'MDS 2017',linestyle = '')
                #companion frequebcies across period bins
                logperiod_day_to_logsmaxes_AU =  lambda x :  np.log10((((((10.0**x)*24*60*60)**2) * 6.67e-11 * average_pands / (4*np.pi**2))**0.333)/m_to_AU)  
                if (upper_limit == 1.3 and lower_limit == 0.7): 
                    obs_periods = np.linspace(3.5,7.5,num = 5)
                    obs_y = np.array([6,7,9,9,10])+np.array([18,27,31,23,21])
                    obs_y_error = np.sqrt(obs_y)
                else:
                    obs_periods =np.array([1,3,5,7])
                    obs_y = np.array([0.16, 0.24, 0.21, 0.12])
                    obs_y_error = np.array([0.05, 0.08, 0.07, 0.04])
                obs_smaxes = logperiod_day_to_logsmaxes_AU(obs_periods)  
                # obs_norm_factor = np.interp(obs_smaxes[np.argmax(obs_y)],x[0],y[0])/np.max(obs_y)
                obs_norm_factor = np.max(y[0][x[0]>np.log10(20)+0.2])/np.max(obs_y)
                ax1.errorbar(obs_smaxes,obs_y*obs_norm_factor,yerr=obs_y_error*obs_norm_factor,xerr = 0.5*np.ones_like(len(obs_smaxes)),marker = 'o',capsize = 5,color = 'black',label = 'MDS 2017',linestyle = '')
            ax1.legend(fontsize=14, labelspacing=0)
        elif which_plot == 'Multiplicity':
            for i in range(0,len(Filenames)):
                plt.plot(x[i],y[i],label = labels[i],color=colors[i])
                if not ( (labels[i]=='') or (labels[min(i+1,len(Filenames)-1)]=='')): #(labels[i]=='Fiducial' and ('' in labels) ) ):
                    plt.fill_between(x[i],np.array(y[i],dtype = np.float32)+error[i],np.array(y[i],dtype = np.float32)-error[i],alpha = 0.2, color=colors[i])
            observation_mass_center = [0.0875,0.205,0.1125,0.225,0.45,1,0.875,1.125,1.175,2,4.5,6.5,12.5]
            observation_mass_width = [0.0075,0.045,0.0375,0.075,0.15,0.25,0.125,0.125,0.325,0.4,1.5,1.5,4.5]
            observation_MF = [0.19,0.20,0.19,0.23,0.3,np.nan,0.42,0.5,0.47,0.68,0.81,0.89,0.93]
            observation_MF_err = [0.07,0.04,0.03,0.02,0.02,np.nan,0.03,0.04,0.03,0.07,0.06,0.05,0.04]
            observation_CF = [0.19,0.20,0.21,0.27,0.38,0.60,np.nan,np.nan,0.62,0.99,1.28,1.55,1.8]
            observation_CF_err = [0.07,0.04,0.03,0.03,0.03,0.04,np.nan,np.nan,0.04,0.13,0.17,0.24,0.3]
            if multiplicity == 'MF':
                for i in range(len(observation_mass_center)):
                    if i == 0:
                        temp_label = 'Observations'
                    else:
                        temp_label = None
                    plt.errorbar(np.log10(observation_mass_center[i]),observation_MF[i],yerr = observation_MF_err[i],xerr = [[np.log10(observation_mass_center[i])-np.log10(observation_mass_center[i]-observation_mass_width[i])],[np.log10(observation_mass_center[i]+observation_mass_width[i])-np.log10(observation_mass_center[i])]],marker = 'o',capsize = 5,color = 'black',label = temp_label,ls='none')
                plt.ylim([-0.01,1.01])
            elif multiplicity == 'CF':
                for i in range(len(observation_mass_center)):
                    if i == 0:
                        temp_label = 'Observations'
                    else:
                        temp_label = None
                    plt.errorbar(np.log10(observation_mass_center[i]),observation_CF[i],yerr = observation_CF_err[i],xerr = [[np.log10(observation_mass_center[i])-np.log10(observation_mass_center[i]-observation_mass_width[i])],[np.log10(observation_mass_center[i]+observation_mass_width[i])-np.log10(observation_mass_center[i])]],marker = 'o',capsize = 5,color = 'black',label = temp_label,ls='none')
                plt.ylim([-0.01,3.01])
            if (multiplicity == 'CF') or  (multiplicity == 'MF'):   
                xlims = plt.xlim(); ylims = plt.ylim()
                plt.fill_between(np.linspace(xlims[0],0,20),ylims[1],ylims[0],color = 'black',alpha = 0.2)
                plt.xlim(xlims); plt.ylim(ylims);  plt.xlabel('Log Mass [$\mathrm{M_\odot}$]')
            plt.legend(fontsize=14)
        elif which_plot == 'Multiplicity Time Evolution':
            plt.figure(figsize = (6,6))
            for i in range(len(Files)):
                if time_plot == 'consistent mass':
                    plt.plot(times[i],cons_fracs[i],label = labels[i], color=colors[i])
                    if not ( (labels[i]=='') or (labels[min(i+1,len(Files)-1)]=='')):
                        plt.fill_between(times[i], cons_fracs[i]-cons_fracs_err[i],y2=cons_fracs[i]+cons_fracs_err[i], alpha=0.3, color=colors[i])
                elif time_plot == 'all':
                    plt.plot(times[i],fractions[i],label = labels[i], color=colors[i])
                    if not ( (labels[i]=='') or (labels[min(i+1,len(Files)-1)]=='')):
                        plt.fill_between(times[i], fractions[i]-fractions_err[i],y2=fractions[i]+fractions_err[i], alpha=0.3, color=colors[i])
            plt.text(0.01,0.03,'Primary Mass = '+str(lower_limit)+' - '+str(upper_limit)+ r' $\mathrm{M_\odot}$',transform = plt.gca().transAxes,horizontalalignment = 'left',fontsize=14)
            if time_norm == 'Myr':
                plt.xlabel('Time [Myr]')
            elif time_norm == 'tff':
                plt.xlabel(r'Time [$t_\mathrm{ff}$]')
            elif time_norm == 'atff':
                plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
            if multiplicity == 'MF':
                plt.ylabel('Multiplicity Fraction')
                plt.ylim(bottom = 0,top = 1.01)
            if multiplicity == 'CF':
                plt.ylabel('Companion Frequency')
            plt.legend(fontsize=14)
        elif which_plot == 'YSO Multiplicity':
            if description is not None:
                save = True
                output_dir = description
                new_file = output_dir+'/YSO'
                mkdir_p(new_file)
            else:
                save = False
            for i in range(len(Files)):
                plt.plot(times[i],fractions[i],label = labels[i], color=colors[i])
                plt.fill_between(times[i], fractions[i]-fraction_err[i],y2=fractions[i]+fraction_err[i], alpha=0.3, color=colors[i])
            if time_norm == 'Myr':
                plt.xlabel('Time [Myr]')
            elif time_norm == 'tff':
                plt.xlabel(r'Time [$t_\mathrm{ff}$]')
            elif time_norm == 'atff':
                plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
            plt.ylabel('YSO Multiplicity Fraction')
            plt.fill_betweenx(np.linspace(0.35,0.5,100),0,max(list(flatten(times))),color = 'orange',alpha = 0.3)
            plt.fill_betweenx(np.linspace(0.3,0.4,100),0,max(list(flatten(times))),color = 'black',alpha = 0.3)
            plt.fill_betweenx(np.linspace(0.25,0.15,100),0,max(list(flatten(times))),color = 'purple',alpha = 0.3)
            plt.text(0.1,0.45,'Class 0 Perseus',fontsize = 16)
            plt.text(0.1,0.32,'Class 0 Orion',fontsize = 16)
            plt.text(0.1,0.2,'Class 1 Orion',fontsize = 16)
            plt.legend(fontsize=14)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
            if save == True:
                plt.savefig(new_file+'/YSO_Multiplicity_'+description+'.png',dpi = 150, bbox_inches='tight')
            plt.figure(figsize = (6,6))
            for i in range(len(Files)):
                plt.plot(times[i],nos[i],label = labels[i], color=colors[i])
            plt.ylabel('Number of YSOs')
            if time_norm == 'Myr':
                plt.xlabel('Time [Myr]')
            elif time_norm == 'tff':
                plt.xlabel(r'Time [$t_\mathrm{ff}$]')
            elif time_norm == 'atff':
                plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
            plt.legend(fontsize=14)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
            if save == True:
                plt.savefig(new_file+'/YSO_Number_'+description+'.png',dpi = 150, bbox_inches='tight')
            plt.figure(figsize = (6,6))
            for i in range(len(Files)):
                plt.plot(times[i],avg_mass[i],label = labels[i], color=colors[i])
            plt.ylabel('Average Mass of YSOs')           
            if time_norm == 'Myr':
                plt.xlabel('Time [Myr]')
            elif time_norm == 'tff':
                plt.xlabel(r'Time [$t_\mathrm{ff}$]')
            elif time_norm == 'atff':
                plt.xlabel(r'Time [$\sqrt{\alpha}t_\mathrm{ff}$]')
            plt.legend(fontsize=14)
            adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14)
            if save == True:
                plt.savefig(new_file+'/YSO_Mass_'+description+'.png',dpi = 150, bbox_inches='tight')
        else:
            if len(offsets)>1:
                while np.ptp(y)<20*np.max(np.abs(np.diff(offsets))):
                    offsets=offsets*0.5
            for i in range(0,len(Filenames)):
                plt.step(x[i]-offsets[i],y[i]-offsets[i],label = labels[i],color=colors[i])
            if which_plot == 'Angle':
                x_pected = np.linspace(0,180,31)
                y_pected = max(flatten(y))*np.sin(x_pected*np.pi/180)
                plt.plot(x_pected,y_pected,label = 'Random',linestyle = ':', color='k')
            plt.legend(fontsize=14)
        adjust_font(fig=plt.gcf(), ax_fontsize=14, labelfontsize=14,lgnd_handle_size=14,adjust_ticks=adjust_ticks)
        if log == True:
            plt.yscale('log')

