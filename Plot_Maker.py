import pickle
from STARFORGE_Multiplicity_Analyzer import load_files,system_initialization,Plots,Multi_Plot,star_system,Plots_key,mkdir_p
from get_sink_data import sinkdata
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from errno import EEXIST
from os import makedirs,path

alpha_filenames = ['M2e4_C_M_J_RT_W_alpha1_2e7','M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_alpha4_2e7']
alpha_labels = [r'$\alpha_\mathrm{turb}=1$',r'$\alpha_\mathrm{turb}=2$',r'$\alpha_\mathrm{turb}=4$']
sigma_filenames = ['M2e4_C_M_J_RT_W_R3_2e7','M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_R30_2e7']
sigma_labels = [r'$\Sigma = 6.3\,M_\mathrm{\odot}/\mathrm{pc}^2$',r'$\Sigma = 63\,M_\mathrm{\odot}/\mathrm{pc}^2$', r'$\Sigma = 630\,M_\mathrm{\odot}/\mathrm{pc}^2$']
BOX_filenames = ['M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_2e7_BOX','M2e4_C_M_J_RT_W_nodriving_2e7_BOX']
BOX_labels = [r'Sphere', r'Box', r'Box, decaying']
metal_filenames = ['M2e4_C_M_J_RT_W_2e7_alt','M2e4_C_M_J_RT_W_Zx01_2e7','M2e4_C_M_J_RT_W_Zx001_2e7']
metal_labels = [r'$\mathrm{Z/Z_\odot}=1$',r'$\mathrm{Z/Z_\odot}=0.1$',r'$\mathrm{Z/Z_\odot}=0.01$']
mu_filenames = ['M2e4_C_M_J_RT_W_2e7_alt','M2e4_C_M_J_RT_W_hiB_2e7','M2e4_C_M_J_RT_W_vhiB_2e7'] 
mu_labes = [r'$\mu=4.2$',r'$\mu=1.3$',r'$\mu=0.42$']
ISRF_filenames = ['M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_ISRFx10_2e7','M2e4_C_M_J_RT_W_ISRFx100_2e7']
ISRF_labels = ['Solar-circle ISRF', '10x ISRF', '100x ISRF']
alt_filenames = ['M2e4_C_M_J_RT_W_2e7_alt','M2e4_C_M_J_RT_W_2e7_alt1','M2e4_C_M_J_RT_W_2e7_alt2']
alt_labels = [r'Random seed = 42',r'Random seed = 1',r'Random seed = 2']

datafolder='C:\Work\Projects\GMC Sim\Analyze\sinkdata'

def all_plots(orig_filenames,description,labels,adaptive_bin_no = 5):
    Filenames = orig_filenames.copy()
    if datafolder!='':
        for i,fname in enumerate(Filenames):
            Filenames[i] = datafolder+'/'+fname
    Files = load_files(Filenames)
    Systems = []
    for i in tqdm(range(len(Files)),position = 0,desc = 'Loading the Systems'):
         Systems.append(system_initialization(Files[i] ,Filenames[i],read_in_result=True))
    
    output_dir = description
    mkdir_p(output_dir)
    
    #Multi_Plot figures
    plt.figure(figsize = (6,6))
    Multi_Plot('Multiplicity',Systems,Files,Filenames,multiplicity='Fraction')
    plt.savefig(description+'/Multiplicity_Fraction_'+description+'.png',dpi = 150)
    plt.figure(figsize = (6,6))
    Multi_Plot('Multiplicity',Systems,Files,Filenames,multiplicity='Frequency')
    plt.savefig(description+'/Companion_Frequency_'+description+'.png',dpi = 150)
    plt.figure(figsize = (6,6))
    Multi_Plot('Semi Major Axis',Systems,Files,Filenames,upper_limit=13000000,lower_limit=0)
    plt.savefig(description+'/Semi_Major_Axis_all_'+description+'.png',dpi = 150)
    plt.figure(figsize = (6,6))
    Multi_Plot('Semi Major Axis',Systems,Files,Filenames,upper_limit=1.3,lower_limit=0.7)
    plt.savefig(description+'/Semi_Major_Axis_Solar_'+description+'.png',dpi = 150)
    plt.figure(figsize = (6,6))
    Multi_Plot('Mass Ratio',Systems,Files,Filenames,upper_limit=13000000,lower_limit=0)
    plt.savefig(description+'/Mass_Ratio_all_'+description+'.png',dpi = 150)
    plt.figure(figsize = (6,6))
    Multi_Plot('Mass Ratio',Systems,Files,Filenames,upper_limit=1.3,lower_limit=0.7)
    plt.savefig(description+'/Mass_Ratio_Solar_'+description+'.png',dpi = 150)
    
    #Random Sampling files
    new_file = output_dir+'/Random_Sampling'
    mkdir_p(new_file)
    for n in tqdm(range(len(Files)),position = 0,desc = 'Random Sampling'):
        plt.figure(figsize = (6,6))
        Plots('Mass Ratio',Systems[n][-1],Files[n],Filenames[n],Master_File=Systems[n],compare=True,snapshot = -1,log = False,bins = np.linspace(0,1,11))
        plt.savefig(new_file+'/Random_Sampling_'+orig_filenames[n]+'.png',dpi = 150)
    
    #System Mass files
    new_file = output_dir+'/System_Mass_Dist'
    mkdir_p(new_file)
    for n in tqdm(range(len(Files)),position = 0,desc = 'System Mass Dist'):
        plt.figure(figsize = (6,6))
        Plots('System Mass',Systems[n][-1],Files[n],Filenames[n],Master_File=Systems[n],compare=True,snapshot = -1,log = False)
        plt.savefig(new_file+'/System_Mass_'+orig_filenames[n]+'.png',dpi = 150)
        
    #Primary Mass files
    new_file = output_dir+'/Primary_Mass_Dist'
    mkdir_p(new_file)
    for n in tqdm(range(len(Files)),position = 0,desc = 'Primary Mass Dist'):
        plt.figure(figsize = (6,6))
        Plots('Primary Mass',Systems[n][-1],Files[n],Filenames[n],Master_File=Systems[n],compare=True,snapshot = -1,log = False)
        plt.savefig(new_file+'/Primary_Mass_'+orig_filenames[n]+'.png',dpi = 150)    
        
    #Formation Density vs Multiplicity Plots
    adaptive_nos = []
    for i in Systems:
        no_of_solar = 0
        for j in i[-1]:
            if 0.7<=j.primary<=1.3:
                no_of_solar += 1
        adaptive_nos.append(int((no_of_solar/adaptive_bin_no)))
    new_file = output_dir+'/Density_vs_Multiplicity'
    mkdir_p(new_file)
    plt.figure(figsize = (6,6))
    Multi_Plot('Multiplicity vs Formation',Systems,Files,Filenames,adaptive_no=adaptive_nos,x_axis='density')
    plt.savefig(new_file+'/volume_density_multiplicity_fraction_'+description+'.png',dpi = 150)
    plt.figure(figsize = (6,6))
    Multi_Plot('Multiplicity vs Formation',Systems,Files,Filenames,adaptive_no=adaptive_nos,x_axis='mass density')
    plt.savefig(new_file+'/mass_density_multiplicity_fraction_'+description+'.png',dpi = 150)
    plt.figure(figsize = (6,6))
    Multi_Plot('Multiplicity vs Formation',Systems,Files,Filenames,adaptive_no=adaptive_nos,x_axis='density',multiplicity='Frequency')
    plt.savefig(new_file+'/volume_density_companion_frequency_'+description+'.png',dpi = 150)
    plt.figure(figsize = (6,6))
    Multi_Plot('Multiplicity vs Formation',Systems,Files,Filenames,adaptive_no=adaptive_nos,x_axis='mass density',multiplicity='Frequency')
    plt.savefig(new_file+'/mass_density_companion_frequency_'+description+'.png',dpi = 150)
    
    #YSO
    plt.figure(figsize = (6,6))
    Multi_Plot('YSO Multiplicity',Systems,Files,Filenames,time_norm='afft',rolling_avg=True,description=description)
    
    #Multiplicity Lifetime Evolution
    new_file = output_dir+'/Multiplicity_Lifetime_Evolution'
    for i in Filenames:
        mkdir_p(new_file+'/'+i)
    for i in tqdm(range(len(Systems)),position = 0,desc = 'Multiplicity Fraction Lifetime Tracking'):
        plt.figure(figsize = (6,6))
        Plots('Multiplicity Lifetime Evolution',Systems[i][-1],Files[i],Filenames[i],Systems[i],adaptive_no=adaptive_nos[i],description=description,rolling_avg=True,multiplicity='Fraction')
    for i in tqdm(range(len(Systems)),position = 0,desc = 'Companion Frequency Lifetime Tracking'):
        plt.figure(figsize = (6,6))
        Plots('Multiplicity Lifetime Evolution',Systems[i][-1],Files[i],Filenames[i],Systems[i],adaptive_no=adaptive_nos[i],description=description,rolling_avg=True,multiplicity='Frequency')
        
all_plots(alpha_filenames,'alpha',alpha_labels)
#all_plots(sigma_filenames,'sigma',sigma_labels)
#all_plots(BOX_filenames,'BOX',BOX_labels)
#all_plots(metal_filenames,'metal',metal_labels)
#all_plots(mu_filenames,'magnetic',mu_labes)
#all_plots(ISRF_filenames,'ISRF',ISRF_labels)
#all_plots(alt_filenames,'realizations',alt_labels)





