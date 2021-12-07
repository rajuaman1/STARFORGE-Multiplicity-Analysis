import pickle
from STARFORGE_Multiplicity_Analyzer import load_files,system_initialization,Plots,Multi_Plot,star_system,Plots_key,mkdir_p,system_creation,flatten,Multiplicity_One_Snap_Plots_Filters,set_colors_and_styles, Seperation_Tracking,Primordial_separation_distribution
from get_sink_data import sinkdata
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from errno import EEXIST
from os import makedirs,path
import time


class Timer():  
    def __init__(self):
        self.start_time = time.perf_counter()
        self.dt_dict={}
    
    def start(self):
        self.start_time= time.perf_counter()

    def dt(self,restart=True,text='',quiet=True):
        dt = time.perf_counter() - self.start_time
        if text=='':
            self.dt_dict[len(self.dt_dict)] = dt
            if not quiet: print("Elapsed time: %4.2f seconds"%(dt))
        else:
            self.dt_dict[text] = dt
            if not quiet: print('\n'+text+" took %4.2f seconds"%(dt))
        if restart:
            self.start_time= time.perf_counter()
            
    def list_times(self):
        print('\n'); print('Elapsed time table:')
        t_sum=0
        for key in self.dt_dict.keys():
            print('\t'+key+' : %4.2f s'%(self.dt_dict[key]))
            t_sum += self.dt_dict[key]
        print('Total time  : %4.2f s'%(t_sum))

alpha_filenames = ['M2e4_C_M_J_RT_W_alpha1_2e7','M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_alpha4_2e7']
alpha_labels = [r'$\alpha_\mathrm{turb}=1$',r'$\alpha_\mathrm{turb}=2$',r'$\alpha_\mathrm{turb}=4$']
sigma_filenames = ['M2e4_C_M_J_RT_W_R30_2e7','M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_R3_2e7']
sigma_labels = [r'$\Sigma = 6.3\,M_\mathrm{\odot}/\mathrm{pc}^2$',r'$\Sigma = 63\,M_\mathrm{\odot}/\mathrm{pc}^2$', r'$\Sigma = 630\,M_\mathrm{\odot}/\mathrm{pc}^2$']
BOX_filenames = ['M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_2e7_BOX','M2e4_C_M_J_RT_W_nodriving_2e7_BOX']
BOX_labels = [r'Sphere', r'Box', r'Box, decaying']
metal_filenames = ['M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_Zx01_2e7','M2e4_C_M_J_RT_W_Zx001_2e7']
metal_labels = [r'$\mathrm{Z/Z_\odot}=1$',r'$\mathrm{Z/Z_\odot}=0.1$',r'$\mathrm{Z/Z_\odot}=0.01$']
mu_filenames = ['M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_hiB_2e7','M2e4_C_M_J_RT_W_vhiB_2e7'] 
mu_labes = [r'$\mu=4.2$',r'$\mu=1.3$',r'$\mu=0.42$']
ISRF_filenames = ['M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_ISRFx10_2e7','M2e4_C_M_J_RT_W_ISRFx100_2e7']
ISRF_labels = ['Solar-circle ISRF', '10x ISRF', '100x ISRF']
alt_filenames = ['M2e4_C_M_J_RT_W_2e7_alt','M2e4_C_M_J_RT_W_2e7_alt1','M2e4_C_M_J_RT_W_2e7_alt2']
alt_labels = [r'Random seed = 42',r'Random seed = 1',r'Random seed = 2']
plots_to_do = ['Multiplicity Lifetime Evolution', 'Multiplicity Time Evolution','YSO','Formation Density vs Multiplicity Plots','Multiplicity Filters','Single Plots', 'Multi_Plot figures' ]

datafolder='D:\Work\Projects\GMC Sim\Analyze\sinkdata' #use '' if in the same directory as script

def redo_system_assignment(filename,datafolder='',seperation_param=None, no_subdivision_for_last_snaps=10, redo_all=False,L = None):
    file_path = filename
    if datafolder !='': file_path = datafolder + '/' + filename
    file = load_files(file_path)[0]
    output = system_initialization(file,file_path,read_in_result=False,full_assignment= True,seperation_param=seperation_param,no_subdivision_for_last_snaps=no_subdivision_for_last_snaps,redo_all=redo_all,L = L)
    #Saving to file
    outfilename=filename+'_Systems'
    if datafolder !='': outfilename = datafolder + '/' + outfilename
    outfile = open(outfilename,'wb')
    pickle.dump(output,outfile)
    outfile.close()

def all_plots(orig_filenames,description,labels,bins = None,adaptive_bin_no = 5,read_in_result=True,Snapshots = None,log = False,target_age = 0.5,min_age = 0,all_companions = True,filters = [None],avg_filter_snaps_no = 10,q_filt_min = 0.1,time_filt_min = 0.1,normalized = False,norm_no = 100,time_plot = 'consistent mass',rolling_avg=True,rolling_window_Myr=0.1,time_norm = 'tff',zero = 'Formation',filter_in_class = False, colors=None, plots_to_do=['All']):
    Filenames = orig_filenames.copy()
    print(Filenames)
    timer = Timer()
    timer.start()
    if datafolder!='':
        for i,fname in enumerate(Filenames):
            Filenames[i] = datafolder+'/'+fname
    Files = load_files(Filenames)
    Systems = []
    for i in tqdm(range(len(Files)),position = 0,desc = 'Loading the Systems'):
         Systems.append(system_initialization(Files[i] ,Filenames[i],read_in_result=read_in_result,full_assignment = True))
    output_dir = description
    mkdir_p(output_dir)
    timer.dt(text='File initialization')
    
    if plots_to_do[0]=='All' or 'Primordial separation' in plots_to_do:
        for n in tqdm(range(len(Files)),position = 0,desc = 'Making primordial separation distribution'):
            Primordial_separation_distribution(Files[n],Systems[n],upper_limit=50,lower_limit = 5.0, apply_filters=True, outfilename='separation_dist_'+orig_filenames[n]+'_massive')
            Primordial_separation_distribution(Files[n],Systems[n],upper_limit=2.0,lower_limit = 0.1, apply_filters=True, outfilename='separation_dist_'+orig_filenames[n]+'_lowmass')
        timer.dt(text='Primordial separation distributions')
    
    
    # for n in tqdm(range(len(Files)),position = 0,desc = 'Separation tracking'):
    #     Seperation_Tracking(Files[n],Systems[n],rolling_avg = False)
    # timer.dt(text='Separation tracking')
    
    Plot_name = ['Multiplicity','Semi Major Axis','Mass Ratio','Angle']
    
    #Multi_Plot figures
    if plots_to_do[0]=='All' or 'Multi_Plot figures' in plots_to_do:
        for filter_or_not in range(2):
            if filter_or_not == 1:
                filters = ['q_filter','time_filter']
                new_file = output_dir+'/Multi_Plot_Filters'
                mkdir_p(new_file)
            elif filter_or_not == 0:
                filters = ['None']
                new_file = output_dir+'/Multi_Plot'
                mkdir_p(new_file)
            for plot_type in Plot_name:
                if plot_type == 'Multiplicity':
                    for multiplicity in ['MF','CF']:
                        plt.figure(figsize = (6,6));print('\n Making '+multiplicity+' plot...')
                        Multi_Plot(plot_type,Systems,Files,Filenames,multiplicity=multiplicity,labels=labels,bins = bins,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,filter_in_class = filter_in_class, colors=colors)
                        plt.savefig(new_file+'/'+multiplicity+'_'+description+'.png',dpi = 150);plt.close('all') 
                else:
                    for upper_limit,lower_limit in zip([1.3e7,1.3],[0,0.7]):
                        plt.figure(figsize = (6,6))
                        if upper_limit == 1.3e7: print('\n Making '+plot_type+' distribution plot for all stars...');star_type = 'all';
                        elif upper_limit == 1.3: print('\n Making '+plot_type+' distribution plot for Solar-type stars...');star_type = 'solar';
                        Multi_Plot(plot_type,Systems,Files,Filenames,upper_limit=upper_limit,lower_limit=lower_limit,labels=labels,bins = bins,log = log,all_companions = all_companions,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,q_filt_min = q_filt_min,time_filt_min = time_filt_min,normalized = normalized,norm_no = norm_no,filter_in_class = filter_in_class, colors=colors)
                        plt.savefig(new_file+'/'+str(plot_type)+'_'+star_type+'_'+description+'.png',dpi = 150, bbox_inches="tight"); plt.close('all') 
        timer.dt(text='Multi plot figures')
    
    
    if plots_to_do[0]=='All' or 'Single Plots' in plots_to_do:
        print('\nSingle Plots ...')
        if Snapshots == None:
            Snapshots = [[-1]]*len(Filenames)
        Snapshots = list(flatten(Snapshots))
        
        Plot_name = ['Random_Sampling','System_Mass_Dist','Primary_Mass_Dist','Multiplicity_Properties', 'Semi_Major_Axis', 'Angle', 'YSO_Multiplicity','Multiplicity_vs_Formation', 'Multiplicity_Time_Evolution']
        Plot_key = ['Mass Ratio','System Mass','Primary Mass','Multiplicity', 'Semi Major Axis', 'Angle', 'YSO Multiplicity', 'Multiplicity vs Formation', 'Multiplicity Time Evolution']
        
        for plot_type,plot_key in zip(Plot_name,Plot_key):
            new_file = output_dir+'/'+plot_type
            mkdir_p(new_file)
            only_filter = False; plot_intermediate_filters=True
            #filters = ['q_filter','time_filter']; 
            filters = ['Raghavan','time_filter'] 
            #if plot_key == 'Multiplicity vs Formation' :
            multiplicity = 'MF'
            if  plot_key == 'Multiplicity':
                multiplicity = 'Properties'
                filters = [None]; only_filter = True; plot_intermediate_filters=False
            for n in tqdm(range(len(Files)),position = 0,desc = plot_type):
                plt.figure(figsize = (6,6))
                Plots(plot_key,Systems[n],Files[n],Filenames[n],compare=True,snapshot = Snapshots[n],bins = bins,label=labels[n],log = log,all_companions = all_companions,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,target_age=target_age, q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter = only_filter,multiplicity=multiplicity,filter_in_class = filter_in_class,plot_intermediate_filters = plot_intermediate_filters, time_norm='Myr') 
                plt.savefig(new_file+'/'+plot_type+'_'+orig_filenames[n]+'.png',dpi = 150,bbox_inches="tight"); plt.close('all') ; plt.close('all')
                if plot_key in ['Semi Major Axis', 'Angle']:
                    filters = ['q_filter','time_filter']
                    plt.figure(figsize = (6,6))
                    Plots(plot_key,Systems[n],Files[n],Filenames[n],compare=True,snapshot = Snapshots[n],bins = bins,label=labels[n],log = log,all_companions = all_companions,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,target_age=target_age,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter = only_filter,multiplicity=multiplicity,filter_in_class = filter_in_class,upper_limit = 1e4,lower_limit=0,plot_intermediate_filters = plot_intermediate_filters) 
                    plt.savefig(new_file+'/'+plot_type+'_'+orig_filenames[n]+'_all.png',dpi = 150,bbox_inches="tight"); plt.close('all') ; plt.close('all')
                if plot_key in ['Angle', 'Semi Major Axis']:
                    for filelabel,lowlim,uplim in zip(['massive', 'lowmass'],[5,0.1],[50,2.0]):
                        filters = ['q_filter','time_filter']
                        plt.figure(figsize = (6,6))
                        Plots(plot_key,Systems[n],Files[n],Filenames[n],compare=True,snapshot = Snapshots[n],bins = bins,label=labels[n],log = log,all_companions = all_companions,filters = filters,avg_filter_snaps_no = avg_filter_snaps_no,target_age=target_age,q_filt_min = q_filt_min,time_filt_min = time_filt_min,only_filter = only_filter,multiplicity=multiplicity,filter_in_class = filter_in_class,upper_limit = uplim,lower_limit=lowlim,plot_intermediate_filters = plot_intermediate_filters) 
                        #plt.text(0.99,0.9,'Primary Mass = '+str(lowlim)+' - '+str(uplim)+ r' $\mathrm{M_\odot}$',transform = plt.gca().transAxes,horizontalalignment = 'right',fontsize=14)
                        plt.savefig(new_file+'/'+plot_type+'_'+orig_filenames[n]+'_'+filelabel+'.png',dpi = 150,bbox_inches="tight"); plt.close('all') ; plt.close('all')
                    
            timer.dt(text= plot_type+' figures')
    
    if plots_to_do[0]=='All' or 'Multiplicity Filters' in plots_to_do:
        new_file = output_dir+'/Multiplicity_Filters'
        mkdir_p(new_file)
        for i in range(len(Files)):
            for multiplicity in ['MF','CF']:
                plt.figure(figsize = (6,6))
                Multiplicity_One_Snap_Plots_Filters(Systems[i],Files[i],multiplicity = multiplicity,filter_in_class = filter_in_class,include_error=True)
                plt.savefig(new_file+'/'+multiplicity+'_Filters_'+orig_filenames[i]+'.png',dpi = 150,bbox_inches="tight"); plt.close('all') ; plt.close('all')

    #Formation Density vs Multiplicity Plots
    if plots_to_do[0]=='All' or 'Formation Density vs Multiplicity Plots' in plots_to_do:
        adaptive_nos = []
        for i in Systems:
            no_of_solar = 0
            for j in i[-1]:
                if 0.7<=j.primary<=1.3:
                    no_of_solar += 1
            adaptive_nos.append(max(int((no_of_solar/adaptive_bin_no)),1))
        new_file = output_dir+'/Density_vs_Multiplicity'
        mkdir_p(new_file)
        for multiplicity in ['MF','CF']:
            for density in ['density','mass density']:
                plt.figure(figsize = (6,6)); print('\n Making '+multiplicity+' vs formation '+density+' plot...')
                Multi_Plot('Multiplicity vs Formation',Systems,Files,Filenames,adaptive_no=adaptive_nos,x_axis=density,multiplicity=multiplicity,labels=labels,rolling_avg=rolling_avg,rolling_window=rolling_window_Myr,time_norm = time_norm, colors=colors)
                plt.savefig(new_file+'/'+density+'_'+multiplicity+'_'+description+'.png',dpi = 150,bbox_inches="tight"); plt.close('all')
        timer.dt(text='Formation Density vs Multiplicity Plots')
    
    #YSO
    if plots_to_do[0]=='All' or 'YSO' in plots_to_do:
        plt.figure(figsize = (6,6)); print('\n YSO multiplicity plot...')
        Multi_Plot('YSO Multiplicity',Systems,Files,Filenames,description=description,labels=labels,target_age=target_age,min_age=min_age,rolling_avg=rolling_avg,rolling_window=rolling_window_Myr,time_norm = time_norm, colors=colors)
        timer.dt(text='YSO Plot')
    
    #Multiplicity Time Evolution
    if plots_to_do[0]=='All' or 'Multiplicity Time Evolution' in plots_to_do:
        new_file = output_dir+'/Multiplicity_Time_Evolution'
        mkdir_p(new_file)
        plt.figure(figsize = (6,6)); print('\n Multiplicity Time Evolution plot...')
        for plot_type in ['all','consistent mass']:
            for multiplicity_type in ['MF','CF']:
                Multi_Plot('Multiplicity Time Evolution',Systems,Files,Filenames,description=description,labels=labels,time_plot = plot_type,rolling_avg=rolling_avg,rolling_window=rolling_window_Myr,time_norm = time_norm,multiplicity=multiplicity_type, colors=colors)
                plt.savefig(new_file+'/'+str(plot_type)+str(multiplicity_type)+'.png',dpi = 150,bbox_inches="tight"); plt.close('all')
        timer.dt(text='Multiplicity Time Evolution')
    
    #Multiplicity Lifetime Evolution
    if plots_to_do[0]=='All' or 'Multiplicity Lifetime Evolution' in plots_to_do:
        new_file = output_dir+'/Multiplicity_Lifetime_Evolution'
        for i in Filenames:
            mkdir_p(new_file+'/'+path.basename(i))
        for i in tqdm(range(len(Systems)),position = 0,desc = 'Multiplicity Fraction Lifetime Tracking'):
            plt.figure(figsize = (6,6)); 
            Plots('Multiplicity Lifetime Evolution',Systems[i],Files[i],Filenames[i],adaptive_no=adaptive_nos[i],description=description,multiplicity='both',label=labels[i],rolling_avg=rolling_avg,rolling_window_Myr = rolling_window_Myr,time_norm = time_norm,zero = zero)
            plt.close('all') 
        timer.dt(text='MF & CF evolution plots') 
    
    #Print out final timing table
    timer.list_times()

redo_all_main=False
sequential_colors_2, _ = set_colors_and_styles(None, None, 2, dark=True, sequential=True)
sequential_colors_3, _ = set_colors_and_styles(None, None, 3, dark=True, sequential=True)
colors_3, _ = set_colors_and_styles(None, None, 3, dark=True, sequential=False)
#plots_to_do = ['Multiplicity Lifetime Evolution', 'Multiplicity Time Evolution','YSO','Formation Density vs Multiplicity Plots','Multiplicity Filters','Single Plots', 'Multi_Plot figures' ]
#plots_to_do = ['Multiplicity Filters','Single Plots', 'Multi_Plot figures' ]
#plots_to_do = ['YSO' ]
#plots_to_do = ['All' ]
plots_to_do = ['Single Plots', 'Multiplicity Time Evolution' ] 


# alpha_filenames = ['M2e4_C_M_J_RT_W_2e7','M2e4_C_M_J_RT_W_alpha4_2e7']
# alpha_labels = [r'$\alpha_\mathrm{turb}=2$',r'$\alpha_\mathrm{turb}=4$']

# redo_system_assignment('M2e3_C_M_J_RT_W_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_2e7_alt2',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_2e7_alt1',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_2e7_alt',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_Zx01_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_Zx001_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_ISRFx10_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_ISRFx100_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_hiB_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_vhiB_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_nodriving_2e7_BOX',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main,L= 16.1122)
# redo_system_assignment('M2e4_C_M_J_RT_W_2e7_BOX',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main,L= 16.1122)
# redo_system_assignment('M2e4_C_M_J_RT_W_R3_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)
# redo_system_assignment('M2e4_C_M_J_RT_W_R30_2e7',datafolder=datafolder,seperation_param=2, redo_all=redo_all_main)


# all_plots(alt_filenames,'realizations',alt_labels,colors=colors_3, plots_to_do=plots_to_do)
# all_plots(alpha_filenames,'alpha',alpha_labels,colors=sequential_colors_3, plots_to_do=plots_to_do)
# all_plots(metal_filenames,'metal',metal_labels,colors=sequential_colors_3, plots_to_do=plots_to_do)
# all_plots(mu_filenames,'magnetic',mu_labes,colors=sequential_colors_3, plots_to_do=plots_to_do)
# all_plots(ISRF_filenames,'ISRF',ISRF_labels,colors=sequential_colors_3, plots_to_do=plots_to_do)
#all_plots(BOX_filenames,'BOX',BOX_labels,colors=colors_3, plots_to_do=plots_to_do)
all_plots(sigma_filenames,'sigma',sigma_labels,colors=sequential_colors_3, plots_to_do=plots_to_do)





