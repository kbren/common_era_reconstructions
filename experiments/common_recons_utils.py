""" 
    Set of functions to facilitate running reanalysis reconstructions. 
    
    Katie Brennan
    Started: June 2019
"""

import sys,os,copy

import sys
#import LMR_config_greg
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spy
import scipy.stats as stats
import pickle 
import xarray as xr

from time import time
from spharm import Spharmt, getspecindx, regrid

sys.path.insert(1,'/home/disk/kalman2/mkb22/LMR_lite/')
import LMR_lite_utils as LMRlite

sys.path.insert(1,'/home/disk/kalman2/mkb22/pyLMR/')
import LMR_utils as lmr

from LMR_utils import global_hemispheric_means

def get_prior(CFG):
    """
        Loads prior and selects and ensemble, then regrids prior. 
        
        Inputs: 
            CFG = config file information
        Outputs: 
            X_regrid
            X
            Xb_one
            Xbp
    """
    print('loading prior')
    X, Xb_one = LMRlite.load_prior(CFG)
    Xbp = Xb_one - Xb_one.mean(axis=1,keepdims=True)

    # use this for labeling graphics and output files
    prior_id_string = X.prior_datadir.split('/')[-1]
    print('prior string label: ',prior_id_string)
    
    # check if config is set to regrid the prior
    if CFG.prior.regrid_method:
        print('regridding prior...')
        # this function over-writes X, even if return is given a different name
        [X_regrid,Xb_one_new] = LMRlite.prior_regrid(CFG,X,Xb_one,verbose=True)
    else:
        X_regrid.trunc_state_info = X.full_state_info
        
    return X_regrid, Xb_one_new, X, Xb_one, Xbp

def calc_calibration_ratio(YE,OBS,R):
    """Calculates the calibration ratio 
    
        Inputs: 
            YE: Ye values (nobs,nens)
            OBS: Observations (nobs, time) 
            R: Observation error (nobs)
            
        Outputs: 
            mse: 
            varYe: 
            CR: calibration ratio
    """
    nobs = OBS.shape[0]
    
    mse = np.empty(nobs)
    varYe = np.empty(nobs)
    CR = np.empty(nobs) 
    
    Ye_ensmean = np.nanmean(YE,axis=1)

    for ob in range(nobs):
        mse[ob] = np.nanmean((Ye_ensmean[ob]-OBS[ob,:])**2)
        varYe[ob] = np.nanvar(YE[ob,:])
        CR[ob] = mse[ob]/(varYe[ob]+R[ob])
    
    return CR, mse, varYe
                         
def load_recon_pickle(LOC,ens=False,tas=False):
    """ Loads 20th century reconstructions output as a pickle 
        for testing the sensitivity to different observation 
        errors used (R). 
        input: LOC = path to pickle file
    """
    
    recon = pickle.load(open(LOC,'rb'))
    
    if ens is False:
        sic_dict = recon['sic_dict']
        sic_dict_lalo = recon['sic_dict_lalo']
        var_dict = recon['var_dict']
        obs_size = recon['obs_size']
        obs_full = recon['obs_full']
        recon_years_all = recon['recon_years_all']
        R_all = recon['R_all']
        Ye_full = recon['Ye_full']
        if tas is True: 
            tas_lalo = recon['tas_dict_lalo']   
            return sic_dict, sic_dict_lalo, var_dict, obs_full, obs_size, recon_years_all, R_all, Ye_full, tas_lalo
        else: 
            return sic_dict, sic_dict_lalo, var_dict, obs_full, obs_size, recon_years_all, R_all, Ye_full
    
    else:
        sic_dict = recon['sic_dict']
        sic_dict_lalo = recon['sic_dict_lalo']
        sia_full_ens = recon['sia ens dict']
        sie_full_ens = recon['sie ens dict']
        var_dict = recon['var_dict']
        obs_size = recon['obs_size']
        obs_full = recon['obs_full']
        recon_years_all = recon['recon_years_all']
        R_all = recon['R_all']
        Ye_full = recon['Ye_full']
        if tas is True: 
            tas_lalo = recon['tas_dict_lalo']   
            return sic_dict, sic_dict_lalo, sia_full_ens, sie_full_ens, var_dict, obs_full, obs_size, recon_years_all, R_all, Ye_full, tas_lalo
        else: 
            return sic_dict, sic_dict_lalo, sia_full_ens, sie_full_ens, var_dict, obs_full, obs_size, recon_years_all, R_all, Ye_full
        
def load_si_conf_intervals_pickle(LOC):
    
    recon = pickle.load(open(LOC,'rb'))
    
    sic_conf_97_5 = recon['sic_conf_97_5']
    sic_conf_2_5 = recon['sic_conf_2_5']
    
    return sic_conf_97_5, sic_conf_2_5
    
def sat_comp_plot(SAT_TIME, SAT_DATA, RECON_YEARS, COMP_VAR, CORR, CE, ANOM_START, ANOM_END, TITLE): 
    """Plots comparison of reconstructions with satellite data with verifcation
       statistics. 
       inputs: 
           SAT_TIME = years of satellite data (nyears)
           SAT_DATA = satellite data (nyears)
           RECON_YEARS = dictionary of recon years 
           COMP_VAR = dictionary of reconstructions
           CORR = dictionary of correlation coefficients 
           CE = dictionary of correlation coefficients
           ANOM_START = year anomalies start
           ANOM_END = last year included in anomalies
           TITLE = str
    """
    plt.subplots(figsize=(15,10))

    plt.plot(SAT_TIME,SAT_DATA,\
           color='r',label ='Satellite (Fetterer etal 2017)',\
           linestyle='-',linewidth=2)
    plt.plot(RECON_YEARS['GIS'],COMP_VAR['GIS'],\
            color='darkslateblue',label ='GISTEMP',\
            linestyle='-',linewidth=1.5)
    plt.plot(RECON_YEARS['BE'],COMP_VAR['BE'],\
            color='darkorchid',label ='Berkeley Earth',\
            linestyle='-',linewidth=1.5)
    plt.plot(RECON_YEARS['CRU'],COMP_VAR['CRU'],\
            color='b',label ='HadCRUT',\
            linestyle='-',linewidth=1.5)
    # plt.plot(walsh_time,walsh_sia_anom,\
    #         color='k',label ='Walsh et al 2017',\
    #         linestyle='-',linewidth=2.5)

    #Set up text to print ce and r values
    ce_corr_gis_sat = ('(Satellite, LMR-GISTEMP):    R= ' + '{:,.2f}'.format(CORR['GIS']) + ', CE= ' + 
                         '{:,.2f}'.format(CE['GIS']))
    ce_corr_cru_sat = ('(Satellite, LMR-HadCRUT):   R= ' + '{:,.2f}'.format(CORR['CRU']) + ', CE= ' + 
                         '{:,.2f}'.format(CE['CRU']))
    ce_corr_be_sat = ('(Satellite, LMR-BE):              R= ' + '{:,.2f}'.format(CORR['BE']) + ', CE= ' + 
                         '{:,.2f}'.format(CE['BE']))

    anomalies = 'Anomalies centered about '+str(ANOM_START)+'-'+str(ANOM_END)

    plt.gcf().text(0.15, 0.18, ce_corr_gis_sat, fontsize=12)
    plt.gcf().text(0.15, 0.16, ce_corr_cru_sat, fontsize=12)
    plt.gcf().text(0.15, 0.14, ce_corr_be_sat, fontsize=12)

    plt.gcf().text(0.66, 0.14, anomalies, fontsize=12)

    plt.axhline(0, color='grey',linestyle='--',linewidth=1.5)

    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Total sea ice anomalies (M $km^{2}$)', fontsize=14)
    plt.title(TITLE, fontsize=16) 

    min_val = round(SAT_DATA.min())
    max_val = round(SAT_DATA.max())+0.51
    
    plt.yticks(np.arange(min_val, max_val, 0.5), fontsize=14)
    plt.xticks(np.arange(1850, 2020,10), fontsize=14)
    plt.xlim((1975,2020))

    plt.legend(fontsize=12, loc='upper right')
    plt.grid()
    
def sat_walsh_comp_plot(SAT_TIME, SAT_DATA, RECON_YEARS, COMP_VAR, WALSH_TIME, WALSH_DATA, CORR, CE, ANOM_START, ANOM_END, TITLE): 
    """Plots comparison of reconstructions with satellite data with verifcation
       statistics. 
       inputs: 
           SAT_TIME = years of satellite data (nyears)
           SAT_DATA = satellite data (nyears)
           RECON_YEARS = dictionary of recon years 
           COMP_VAR = dictionary of reconstructions
           CORR = dictionary of correlation coefficients 
           CE = dictionary of correlation coefficients
           ANOM_START = year anomalies start
           ANOM_END = last year included in anomalies
           TITLE = str
    """
    plt.subplots(figsize=(15,10))

    plt.plot(SAT_TIME,SAT_DATA,\
           color='r',label ='Satellite (Fetterer etal 2017)',\
           linestyle='-',linewidth=2)
    plt.plot(RECON_YEARS['GIS'],COMP_VAR['GIS'],\
            color='darkslateblue',label ='GISTEMP',\
            linestyle='-',linewidth=1.5)
    plt.plot(RECON_YEARS['BE'],COMP_VAR['BE'],\
            color='darkorchid',label ='Berkeley Earth',\
            linestyle='-',linewidth=1.5)
    plt.plot(RECON_YEARS['CRU'],COMP_VAR['CRU'],\
            color='b',label ='HadCRUT',\
            linestyle='-',linewidth=1.5)
    plt.plot(WALSH_TIME,WALSH_DATA,\
            color='k',label ='Walsh et al 2017',\
            linestyle='-',linewidth=2.5)

    #Set up text to print ce and r values
    ce_corr_gis_sat = ('(Satellite, LMR-GISTEMP):    R= ' + '{:,.2f}'.format(CORR['GIS']) + ', CE= ' + 
                         '{:,.2f}'.format(CE['GIS']))
    ce_corr_cru_sat = ('(Satellite, LMR-HadCRUT):   R= ' + '{:,.2f}'.format(CORR['CRU']) + ', CE= ' + 
                         '{:,.2f}'.format(CE['CRU']))
    ce_corr_be_sat = ('(Satellite, LMR-BE):              R= ' + '{:,.2f}'.format(CORR['BE']) + ', CE= ' + 
                         '{:,.2f}'.format(CE['BE']))

    anomalies = 'Anomalies centered about '+str(ANOM_START)+'-'+str(ANOM_END)

    plt.gcf().text(0.15, 0.18, ce_corr_gis_sat, fontsize=12)
    plt.gcf().text(0.15, 0.16, ce_corr_cru_sat, fontsize=12)
    plt.gcf().text(0.15, 0.14, ce_corr_be_sat, fontsize=12)

    plt.gcf().text(0.66, 0.14, anomalies, fontsize=12)

    plt.axhline(0, color='grey',linestyle='--',linewidth=1.5)

    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Total sea ice anomalies (M $km^{2}$)', fontsize=14)
    plt.title(TITLE, fontsize=16) 

    min_val = round(WALSH_DATA.min())
    max_val = round(WALSH_DATA.max())+0.51
    
    plt.yticks(np.arange(min_val, max_val, 0.5), fontsize=14)
    plt.xticks(np.arange(1850, 2020,10), fontsize=14)
    plt.xlim((1850,2020))

    plt.legend(fontsize=12, loc='upper right')
    plt.grid()
    
def sat_walsh_comp_plot_single(SAT_TIME, SAT_DATA, RECON_YEARS, COMP_VAR, WALSH_TIME, WALSH_DATA, CORR, CE, ANOM_START, ANOM_END, TITLE): 
    """Plots comparison of reconstructions with satellite data with verifcation
       statistics. 
       inputs: 
           SAT_TIME = years of satellite data (nyears)
           SAT_DATA = satellite data (nyears)
           RECON_YEARS = dictionary of recon years 
           COMP_VAR = dictionary of reconstructions
           CORR = dictionary of correlation coefficients 
           CE = dictionary of correlation coefficients
           ANOM_START = year anomalies start
           ANOM_END = last year included in anomalies
           TITLE = str
    """
    plt.subplots(figsize=(15,10))

    plt.plot(SAT_TIME,SAT_DATA,\
           color='r',label ='Satellite (Fetterer etal 2017)',\
           linestyle='-',linewidth=2)
    plt.plot(RECON_YEARS,COMP_VAR,\
            color='b',label ='HadCRUT',\
            linestyle='-',linewidth=1.5)
    plt.plot(WALSH_TIME,WALSH_DATA,\
            color='k',label ='Walsh et al 2017',\
            linestyle='-',linewidth=2.5)

    #Set up text to print ce and r values
    ce_corr_cru_sat = ('(Satellite, LMR-GISTEMP):    R= ' + '{:,.2f}'.format(CORR) + ', CE= ' + 
                         '{:,.2f}'.format(CE))

    anomalies = 'Anomalies centered about '+str(ANOM_START)+'-'+str(ANOM_END)

    plt.gcf().text(0.15, 0.14, ce_corr_cru_sat, fontsize=12)

    plt.gcf().text(0.66, 0.14, anomalies, fontsize=12)

    plt.axhline(0, color='grey',linestyle='--',linewidth=1.5)

    plt.xlabel('Time (years)', fontsize=14)
    plt.ylabel('Total sea ice anomalies (M $km^{2}$)', fontsize=14)
    plt.title(TITLE, fontsize=16) 

    min_val = round(WALSH_DATA.min())
    max_val = round(WALSH_DATA.max())+0.51
    
    plt.yticks(np.arange(min_val, max_val, 0.5), fontsize=14)
    plt.xticks(np.arange(1850, 2020,10), fontsize=14)
    plt.xlim((1850,2020))

    plt.legend(fontsize=12, loc='upper right')
    plt.grid()
        
def R_comp_bar_plot(R1,R2,R3,R4,TITLE,MIN_HEIGHT): 
    """Generates a bar plot for comparing different estimates of R. 
    """
    width = 0.45  # the width of the bars

    # the x locations for the groups
    ind1 = np.arange(len(R1))*2
    ind2 = [x + width for x in ind1]
    ind3 = [x + width for x in ind2]
    ind4 = [x + width for x in ind3]

    fig, ax = plt.subplots(figsize=(10,5))

    rects1 = ax.bar(ind1, R1, width,label='R=1 everywhere', color='royalblue')
    rects2 = ax.bar(ind2, R2, width,label='Calculated R, LB=1', color='g')
    rects3 = ax.bar(ind3, R3, width,label='Calculated R, LB=0.25', color='mediumseagreen')
    rects4 = ax.bar(ind4, R4, width,label='Calculated R, LB=2', color='darkslategrey')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('(R)')
    ax.set_title(TITLE, fontsize=14)
    ax.set_xticks(ind1+0.6)
    ax.set_xticklabels(['GIS', 'BE', 'CRU'],fontsize=16)
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., fontsize=14)
    ax.set_ylim(MIN_HEIGHT,1.0)

    autolabel(rects1, ax, "center")
    autolabel(rects2, ax, "center")
    autolabel(rects3, ax, "center")
    autolabel(rects4, ax, "center")

    fig.tight_layout()
    
def autolabel(rects, ax, xpos='center'):
    """ Copied from the interwebs for labeling bar plots. 
    
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')
        
def load_annual_satellite_anom(ANOM_END): 
    """Loads annual satellite data and finds anomalies that start at 1979 and go 
       to ANOM_END. 
    """
    # Import satellite data Fetterer v3: 
    fet_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Observations/Fetterer_v3/'
    #fet_file = 'Fetterer_data_v3_annual_1978_2017.npz'
    fet_file = 'Fetterer_data_v3_annual_78_17.npz'

    # Load annual data
    fet_loc = fet_directory + fet_file
    fet_data = np.load(fet_loc)

    fet_sie = fet_data['si_extent'][1:]
    fet_sia = fet_data['si_area'][1:]
    fet_time = fet_data['time'][1:]

    # Find anomalies: 

    # Calculate mean 
    fet_anom_cent_sia = np.nanmean(fet_sia[np.where(fet_time<=ANOM_END)],axis=0)
    fet_anom_cent_sie = np.nanmean(fet_sie[np.where(fet_time<=ANOM_END)],axis=0)

    # Find anomalies:  
    fet_sia_anom = fet_sia - fet_anom_cent_sia
    fet_sie_anom = fet_sie - fet_anom_cent_sie
    
    return fet_sia_anom, fet_sie_anom, fet_time

def load_monthly_satellite_anom(ANOM_END): 
    """Loads monthly satellite data and finds anomalies that start at 1979 and go 
       to ANOM_END. 
    """
    # Import satellite data Fetterer v3: 
    fet_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Observations/Fetterer_v3/'
    fet_file = 'Fetterer_data_v3_monthly_79_17.npz'
    fet_file_pkl = 'Fetterer_data_v3_monthly_78_18.pkl'

    # Load annual data
    fet_loc = fet_directory + fet_file
    fet_data = np.load(fet_loc)

    fet_sie = fet_data['si_extent_mo']
    fet_sia = fet_data['si_area_mo']
    fet_time = np.arange(1979,2018,1)

    # Find anomalies: 

    # Calculate mean 
    fet_anom_cent_sia = np.nanmean(fet_sia[np.where(fet_time<=2013)[0],:],axis=0)
    fet_anom_cent_sie = np.nanmean(fet_sie[np.where(fet_time<=2013)[0],:],axis=0)

    # Find anomalies:  
    fet_sia_anom = fet_sia - fet_anom_cent_sia
    fet_sie_anom = fet_sie - fet_anom_cent_sie
    
    return fet_sia_anom, fet_sie_anom, fet_time

def load_annual_satellite(): 
    """Loads annual satellite data and finds anomalies that start at 1979 and go 
       to ANOM_END. 
    """
    # Import satellite data Fetterer v3: 
    fet_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Observations/Fetterer_v3/'
    #fet_file = 'Fetterer_data_v3_annual_1978_2017.npz'
    fet_file = 'Fetterer_data_v3_annual_78_17.npz'

    # Load annual data
    fet_loc = fet_directory + fet_file
    fet_data = np.load(fet_loc)

    fet_sie = fet_data['si_extent']
    fet_sia = fet_data['si_area']
    fet_time = fet_data['time']
    
    return fet_sia, fet_sie, fet_time

def load_monthly_satellite(): 
    """Loads monthly satellite data and finds anomalies that start at 1979 and go 
       to ANOM_END. 
    """
    # Import satellite data Fetterer v3: 
    fet_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Observations/Fetterer_v3/'
    fet_file = 'Fetterer_data_v3_monthly_79_17.npz'
    fet_file_pkl = 'Fetterer_data_v3_monthly_78_18.pkl'

    # Load annual data
    fet_loc = fet_directory + fet_file
    fet_data = np.load(fet_loc)

    fet_sie = fet_data['si_extent_mo']
    fet_sia = fet_data['si_area_mo']
    fet_time = np.arange(1979,2018,1)
    
    return fet_sia, fet_sie, fet_time

# def load_month_satellite(ANOM_END): 
#     """Loads annual satellite data and finds anomalies that start at 1979 and go 
#        to ANOM_END. 
#     """
#     # Import satellite data Fetterer v3: 
#     fet_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Observations/Fetterer_v3/'
#     fet_mo_file = 'Fetterer_data_v3_monthly.pkl'

#     # Load monthly data    
#     fet_mo_data = pd.read_pickle(fet_directory + fet_mo_file)
#     fet_mo_data_79 = fet_mo_data.iloc[2:]

#     fet_mo_data_79[fet_mo_data_79[' extent']<-10] = np.nan
#     fet_mo_data_79[fet_mo_data_79['   area']<-10] = np.nan
    
#     #fet_mo = fet_mo_data.loc[(fet_mo_data[' mo']==MONTH)]
#     # fet_mo = fet_mo_data_79.drop(columns=[' region','    data-type']).loc[(fet_mo_data_79[' mo']>=start_mo) & (fet_mo_data_79['     mo']<=end_mo)]
#     # fet_mo['   area'][fet_mo['   area']<-10] = np.nan

#     # Find anomalies: 
#     # Calculate mean 
#     fet_mo_mean = fet_mo.loc[fet_mo['year']<=ANOM_END].mean()

#     # Find anomalies:  
#     fet_mo_sie_anom = fet_mo[' extent']-fet_mo_mean[' extent']
#     fet_mo_sia_anom = fet_mo['   area']-fet_mo_mean['   area']
    
#     return fet_mo_sia_anom, fet_mo_sie_anom, fet_time

def find_anomalies(TIME, VAR, ANOM_START, ANOM_END, mean=False):
    """Finds anomalies between (and including) ANOM_START and ANOM_END. 
       inputs: 
       TIME = array of years (time) 
       VAR = variable to take anomalies of (time)
       ANOM_START = year anomaly period starts
       ANOM_END = year anomaly period ends (included) 
    """
   
    gt = np.where(TIME>=ANOM_START)
    lt = np.where(TIME<=ANOM_END)
    
    VAR_mean = np.nanmean(VAR[gt[0].min():lt[0].max()],axis=0)
    VAR_anom = VAR - VAR_mean
    
    if mean is True: 
        return VAR_anom, VAR_mean
    else: 
        return VAR_anom

def calc_extent(SIC, GRID, CUTOFF):
#    sie_nhtot = {}

    #for ref_dset in dset_chosen: 
    sie_lalo = SIC

    sie_lalo[sie_lalo<=CUTOFF] = 0.0
    sie_lalo[sie_lalo>CUTOFF] = 100.0

    _,sie_nhtot,_ = lmr.global_hemispheric_means(sie_lalo,GRID.lat[:, 0])
    
    return sie_nhtot, sie_lalo

def load_annual_walsh():
    walsh_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Walsh2016/walsh_comparison/'

    walsh_sie_file_an = 'Walsh_annual_sie_km2.npz'
    walsh_sia_file_an = 'Walsh_annual_sia_km2.npz'

    walsh_sie_data_an = np.load(walsh_directory + walsh_sie_file_an)
    walsh_sia_data_an = np.load(walsh_directory + walsh_sia_file_an)
    
    walsh_sia = walsh_sia_data_an['walsh_nh_annual_area_total']
    walsh_sie = walsh_sie_data_an['walsh_nh_annual_extent_total']
    
    walsh_time = np.arange(1850,2014,1)
    
    return walsh_sia, walsh_sie, walsh_time

def load_monthly_walsh(ANOM_START,ANOM_END):
    walsh_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Walsh2016/walsh_comparison/'

    walsh_sie_file_mo = 'Walsh_monthly_sie_km2.npz'
    walsh_sia_file_mo = 'Walsh_monthly_sia_km2.npz'

    walsh_sie_data_mo = np.load(walsh_directory + walsh_sie_file_mo)
    walsh_sia_data_mo = np.load(walsh_directory + walsh_sia_file_mo)

    walsh_sia = walsh_sia_data_mo['walsh_nh_monthly_area_total']
    walsh_sie = walsh_sie_data_mo['walsh_nh_monthly_extent_total']

    walsh_time = np.arange(1850,2014,1)
    
    return walsh_sia, walsh_sie, walsh_time
    
def find_ce_corr(VAR, REF, REF_TIME, VAR_TIME, START_TIME, END_TIME, detrend=False):
    """Finds the correlation coefficient and coefficient of efficiency between 
       REF and VAR between START_TIME and END_TIME.
       inputs: 
           VAR = test data (1D in time)
           REF = reference data (1D in time) 
           REF_time = reference data time (1D time)
           VAR_TIME = test data time (1D years)
           START_TIME = comparison start year to be included (float)
           END_TIME = last year included in comparison (float)
       
    """
    yr_range_var = np.where((VAR_TIME>=START_TIME)&(VAR_TIME<END_TIME+1))
    yr_range_ref = np.where((REF_TIME>=START_TIME)&(REF_TIME<END_TIME+1))
    
    if detrend is False: 
        ref = REF[yr_range_ref[0]]
        var = VAR[yr_range_var[0]]
    else: 
        ref = spy.detrend(REF[yr_range_ref])
        var = spy.detrend(VAR[yr_range_var])
        
    ce = lmr.coefficient_efficiency(ref,var)
    corr = np.corrcoef(ref,var)[0,1]
    var_ref = np.var(ref)
    var_var = np.var(var)
    
    return ce, corr,var_ref,var_var

# def find_ce_corr_detrended(VAR, REF, REF_TIME, VAR_TIME, START_TIME, END_TIME):
#     """Finds the correlation coefficient and coefficient of efficiency between 
#        REF and VAR between START_TIME and END_TIME.
#        inputs: 
#            VAR = test data (1D in time)
#            REF = reference data (1D in time) 
#            REF_TIME = reference data time (1D in time)
#            VAR_TIME = test data time (1D years)
#            START_TIME = comparison start year to be included (float)
#            END_TIME = last year included in comparison (float)
       
#     """
#     yr_range_var = np.where((np.array(VAR_TIME)>=START_TIME)&(np.array(VAR_TIME)<END_TIME+1))
#     yr_range_ref = np.where((np.array(REF_TIME)>=START_TIME)&(np.array(REF_TIME)<END_TIME+1))
#     ce = lmr.coefficient_efficiency(spy.detrend(REF[yr_range_ref]),spy.detrend(VAR[yr_range_var]))
#     corr = np.corrcoef(spy.detrend(REF[yr_range_ref]),spy.detrend(VAR[yr_range_var]))[0,1]
#     var_ref = np.var(spy.detrend(REF[yr_range_ref]))
#     var_VAR = np.var(spy.detrend(VAR[yr_range_var]))
    
#     return ce, corr, var_ref, var_VAR

def find_trend(VAR, VAR_TIME, START, END, all_output=False):
    """Finds linear trend via regression of a timeseries
    """
    ind = np.where((VAR_TIME>=START)&(VAR_TIME<END+1))

    if all_output is True:
        [slope, intercept, r_value, p_value, std_err] = stats.linregress(VAR_TIME[ind],VAR[ind])
        
        return slope, intercept, r_value, p_value, std_err
    else:
        slope = stats.linregress(VAR_TIME[ind],VAR[ind])[0]
        
        return slope

def overlapping_mean(VAR1, TIME1, VAR2, TIME2, VAR3, TIME3, START, END):
    """Finds mean of overlaping time period of three datasets. 
    """
    ind1 = np.where((TIME1>=START)&(TIME1<END+1))
    ind2 = np.where((TIME2>=START)&(TIME2<END+1))
    ind3 = np.where((TIME3>=START)&(TIME3<END+1))

    all_var = np.zeros((3,ind1[0].shape[0]))

    all_var[0,:] = VAR1[ind1]
    all_var[1,:] = VAR2[ind2]
    all_var[2,:] = VAR3[ind3]

    var_mean = np.nanmean(all_var,axis=0)
    mean_time = np.arange(START,END+1,1)
    
    return mean_time, var_mean 

def sie_iteration_mean(VAR, TIME, NAME_DICT, START, END, DSET, area=False):
    """Finds mean of of different iterations.
       inputs: 
          VAR = dictionary of different iterations
          NAME_DICT = list of keys the mean is taken over (list of strings) 
    """
    niter = len(NAME_DICT)
    # NH surface area in M km^2 from concentration in percentage
    nharea = 2*np.pi*(6380**2)/1e8
    
    time = np.array(TIME[NAME_DICT[0]][DSET])
    ind = np.where((time>=START)&(time<END+1))
    
    all_var = np.zeros((niter,ind[0].shape[0]))
    
    for n in range(niter):
        time = np.array(TIME[NAME_DICT[n]][DSET])
        ind = np.where((time>=START)&(time<END+1))
        
        all_var[n,:] = VAR[NAME_DICT[n]][DSET][ind]*nharea

    var_mean = np.nanmean(all_var,axis=0)
    mean_time = np.arange(START,END+1,1)
    
    return mean_time, var_mean 

def load_recon_sie(PATH,CHOSEN_DSET,ANOM_START,ANOM_END,tas=False):
    if tas is True:
        [sic_dict,
         sic_dict_lalo,
         var_dict,
         obs_full,
         obs_size,
         recon_years_all,
         R_all,
         Ye_full,
         tas_lalo] = load_recon_pickle(PATH,tas=True)
    else:
        [sic_dict,
         sic_dict_lalo,
         var_dict,
         obs_full,
         obs_size,
         recon_years_all,
         R_all,
         Ye_full] = load_recon_pickle(PATH)

    grid_path = '/home/disk/p/mkb22/nobackup/LMR_output/reanalysis_reconstruction_data/'
    grid = pickle.load(open(grid_path+'sic_recon_grid.pkl','rb'))

    sic_anom = {}
    sie_nhtot = {}
    sie_lalo = {}
    sie_anom = {}

    for dataset in CHOSEN_DSET: 
        sic_anom[dataset] = find_anomalies(np.array(np.array(recon_years_all[dataset])), 
                                                      np.squeeze(np.array(sic_dict[dataset])), 
                                                      ANOM_START, ANOM_END)

        #Find extent:
        [sie_nhtot[dataset], 
         sie_lalo[dataset]] = calc_extent(np.squeeze(np.array(sic_dict_lalo[dataset])),grid,15.0) 

        #Find extent anomalies: 
        sie_anom[dataset] = find_anomalies(np.array(np.array(recon_years_all[dataset])), 
                                                          sie_nhtot[dataset], 
                                                          ANOM_START, ANOM_END)
    if tas is True: 
        return sie_nhtot, sic_dict_lalo, sie_anom, recon_years_all, tas_lalo
    else: 
        return sie_nhtot, sic_dict_lalo, sie_anom, recon_years_all

def plot_stats(VAR,TITLE,XVALS):
    plt.figure(figsize=(8,6))

    for i in range(15):
        if XVALS[i] is XVALS[0]:
            col = 'r'
        elif (XVALS[i] is XVALS[0]+1):
            col = 'b'
        else: 
            col='m'
            
        if i is 0:
            lab = name_dict[i+1]
        elif i is 5:
            lab = name_dict[i+1]
        elif i is 10:
            lab = name_dict[i+1]
        else: 
            lab = None
            
        plt.plot(XVALS[i],VAR[name_dict[i+1]], marker='o', 
                 markersize=15, label=lab,color=col)

    if VAR is slope_Rcru:
        plt.axhline(sat_slope,linestyle='--',linewidth=2,color='k',label='satellite')   
    elif VAR is slope_mean:
        plt.axhline(sat_slope,linestyle='--',linewidth=2,color='k',label='satellite') 
    elif VAR is var_Rcru_dt:
        plt.axhline(VAR['sat'],linestyle='--',linewidth=2,color='k',label='satellite') 
    elif VAR is var_sie_detrend_mean:
        plt.axhline(VAR['sat'],linestyle='--',linewidth=2,color='k',label='satellite') 

    plt.legend(bbox_to_anchor=(1.5, 0.75),fontsize=12)
    plt.title(TITLE, fontsize=14)
    plt.show()
    
def plot_stats_4(VAR,VAR_NAMES,TITLE,XVALS,NAME_DICT,INF_FAC,SAT_SLOPE,Y_MAX,Y_MIN,NITER,NTEST,SAVEPATH=None):
    fig, ax = plt.subplots(figsize=(15,11),nrows=2,ncols=2)
    
    for j,V in enumerate(VAR):
        for i in range(NTEST*NITER):
            if (i<NITER):
                col = 'r'
            elif (i >= NITER)&(i<NITER*2):
                col = 'b'
            elif (i >= NITER*2): 
                col='m'
            else:
                col='c'

            if i is 0:
                lab = NAME_DICT[i+1]
            elif i is 5:
                lab = NAME_DICT[i+1]
            elif i is 10:
                lab = NAME_DICT[i+1]
            elif i is 15:
                lab=NAME_DICT[i+1]
            else: 
                lab = None

            plt.subplot(2,2,j+1)
            plt.plot(INF_FAC[i],V[NAME_DICT[i+1]], marker='o', 
                     markersize=15, label=lab,color=col)
            plt.title(TITLE[j], fontsize=14)
            plt.ylim(Y_MIN[j],Y_MAX[j])
            
        print(V)
        if VAR_NAMES[j] is 'slope_Rcru':
            plt.axhline(SAT_SLOPE,linestyle='--',linewidth=2,color='k',label='satellite')   
        elif VAR_NAMES[j] is 'slope_mean':
            plt.axhline(SAT_SLOPE,linestyle='--',linewidth=2,color='k',label='satellite') 
        elif VAR_NAMES[j] is 'var_Rcru_dt':
            plt.axhline(V['sat'],linestyle='--',linewidth=2,color='k',label='satellite') 
        elif VAR_NAMES[j] is 'var_sie_detrend_mean':
            plt.axhline(V['sat'],linestyle='--',linewidth=2,color='k',label='satellite') 

    plt.legend(bbox_to_anchor=(1.45,2.2),fontsize=12)
        #fig.title('Month 3: sat vs. recon')
    
    if SAVEPATH is None: 
        plt.show()
    else:
        plt.savefig(SAVEPATH,bbox_inches='tight')
    
def find_mean_across_iterations(SLOPE,NAME_DICT,NTEST,NITS):
    """Finds the mmean slope across all iterations. 
    
       inputs: 
           SLOPE = dictionary of slopes for each iteration
           NTEST = number of sets of runs to be tested (integer)
           NITS = number of iterations (integer)
           
       outputs: 
           slope_tot = matrix of all slope values (NTEST,NITS)
           slope_mean = mean slope across all iterations (NTEST)
    """
    slope_tot = np.zeros((NTEST,NITS)) 
    
    for j in range(NTEST):
        disp = 1+j*NITS
        for i in range(NITS):
            slope_tot[j,i] = SLOPE[NAME_DICT[i+disp]]
    #         print(name_dict[i+disp[j]])
    #     print("")

    slope_mean = np.nanmean(slope_tot,axis=1)
    
    return slope_tot,slope_mean

def find_best_slope_fit(SLOPE_MEAN,SAT_SLOPE,NTEST):
    """Finds the mmean slope across all iterations. 
    
       inputs: 
           SLOPE_MEAN = mean slope across all iterations (NTEST) 
                        (output from find_mean_across_iterations)
           SAT_SLOPE = slope over the satellite era 
           NTEST = number of sets of runs to be tested (integer) 
           INF_FAC = list of inflation factors being tested
 
       outputs: 
           inf_loc = index of the ideal inflation factor 
    """
    diff_mean = np.zeros(NTEST)

    for i in range(NTEST):
        diff_mean[i] = np.abs(SAT_SLOPE - SLOPE_MEAN[i])
        print(diff_mean[i])
        
    inf_loc = np.where(np.isclose(diff_mean,np.min(diff_mean),atol=10e-6))[0][0]
    
    return inf_loc

def load_monthly_gridded_sat(start_year=1979,end_year=2018):
    sat_num = ['n07_','f08_','f08_','f08_','f11_','f13_','f17_']
    sat_end = ['_v03r01.nc','_v03r01.nc','_dummy.nc','_v03r01.nc','_v03r01.nc','_v03r01.nc','_v03r01.nc']
    sat_path = '/home/disk/eos8/ed/seaice_data/monthly_SIC/monthly_NH/'

    years_all = np.arange(start_year,end_year,1)
    months = np.arange(1,13,1)

    s = 0
    for iy,y in enumerate(years_all):  
        for imo,mo in enumerate(months): 
            if (iy is 8) and (imo is 7):
                end = False
                s = s+1
            elif (iy is 8) and (imo is 11):
                end = False
                s = s+1
            elif (iy is 9) and (imo is 1):
                end = False
                s = s+1
            elif (iy is 13) and (imo is 0):
                end = False
                s = s+1
            elif (iy is 16) and (imo is 9):
                end = False
                s = s+1
            elif (iy is 29) and (imo is 0):
                end = False
                s = s+1
            elif (iy is 38) and (imo is 2):
                end = True 
                break
            else:
                end = False
                s = s   
            filename = 'seaice_conc_monthly_nh_'+sat_num[s]+'%04d%02d'%(y,mo)+sat_end[s]
            data_loc = sat_path+filename
            data1 = xr.open_dataset(data_loc)
            if (iy is 0) and (imo is 0):
                data_sic = data1
            else: 
                data_sic = xr.concat((data_sic,data1),'time')
        if end: 
            break
            
    return data_sic

def load_recon_sie_95(PATH,CHOSEN_DSET,ANOM_START,ANOM_END,full=True,tas=False):
    if tas is True:
        [sic_dict,
         sic_dict_lalo,
         sia_full_ens, 
         sie_full_ens,
         var_dict,
         obs_full,
         obs_size,
         recon_years_all,
         R_all,
         Ye_full,
         tas_dict_lalo] = load_recon_pickle(PATH,ens=True,tas=True)
    else: 
        [sic_dict,
         sic_dict_lalo,
         sia_full_ens, 
         sie_full_ens,
         var_dict,
         obs_full,
         obs_size,
         recon_years_all,
         R_all,
         Ye_full] = load_recon_pickle(PATH,ens=True)

    loc = '/home/disk/p/mkb22/nobackup/LMR_output/reanalysis_reconstruction_data/'
    grid = pickle.load(open(loc +'sic_recon_grid.pkl','rb'))

    sia_lalo = {}
    sia_anom = {}
    sie_anom = {}
    sia_2_5 = {}
    sia_97_5 = {}
    sie_2_5 = {}
    sie_97_5 = {}
    sia_2_5_anom = {}
    sia_97_5_anom = {}
    sie_2_5_anom = {}
    sie_97_5_anom = {}

    for dataset in CHOSEN_DSET: 
        sia = np.nanmean(np.array(sia_full_ens[dataset]),axis=1)        
        sie = np.nanmean(np.array(sie_full_ens[dataset]),axis=1)
        
        sia_anom[dataset], sia_mean = find_anomalies(np.array(np.array(recon_years_all[dataset])), 
                                                      sia,ANOM_START,ANOM_END, mean=True)
        sie_anom[dataset], sie_mean = find_anomalies(np.array(np.array(recon_years_all[dataset])), 
                                                      sie,ANOM_START,ANOM_END, mean=True)
        
        sie_97_5[dataset] = np.percentile(sie_full_ens[dataset],97.5,axis=1)-sie_mean
        sie_2_5[dataset] = np.percentile(sie_full_ens[dataset],2.5,axis=1)-sie_mean
        
        sia_97_5[dataset] = np.percentile(sia_full_ens[dataset],97.5,axis=1)-sia_mean
        sia_2_5[dataset] = np.percentile(sia_full_ens[dataset],2.5,axis=1)-sia_mean

    if full is True: 
        if tas is True: 
            return sia_anom, sia_97_5, sia_2_5, sie_anom, sie_97_5, sie_2_5, recon_years_all, sie_full_ens, tas_dict_lalo
        else: 
            return sia_anom, sia_97_5, sia_2_5, sie_anom, sie_97_5, sie_2_5, recon_years_all, sie_full_ens
    else: 
        if tas is True: 
            return sia_anom, sia_97_5, sia_2_5, sie_anom, sie_97_5, sie_2_5, recon_years_all, tas_dict_lalo
        else: 
            return sia_anom, sia_97_5, sia_2_5, sie_anom, sie_97_5, sie_2_5, recon_years_all
        
def load_recon_grid(): 
    loc = '/home/disk/p/mkb22/nobackup/LMR_output/reanalysis_reconstruction_data/'
    grid = pickle.load(open(loc +'sic_recon_grid.pkl','rb'))
    
    recon_lat = grid.lat[:,0]
    recon_lon = grid.lon[0,:]
    
    return recon_lat, recon_lon

def initialize_regional_dict(regions,nyears,ngrid):
    reg_sie_anom = {}
    reg_sie_lowanom= {}
    reg_sie_highanom= {}
    reg_sic_anom= {}
    reg_sic_lowanom= {}
    reg_sic_highanom= {}

    for reg in regions:
        reg_sie_anom[reg] = np.zeros((nyears,ngrid))
        reg_sie_lowanom[reg]= np.zeros((nyears,ngrid))
        reg_sie_highanom[reg]= np.zeros((nyears,ngrid))
        reg_sic_anom[reg]= np.zeros((nyears,ngrid))
        reg_sic_lowanom[reg]= np.zeros((nyears,ngrid))
        reg_sic_highanom[reg]= np.zeros((nyears,ngrid))
        
    return (reg_sie_anom,reg_sie_lowanom,reg_sie_highanom,
            reg_sic_anom,reg_sic_lowanom,reg_sic_highanom)

def calc_regional_totals(ccsm4_sic, ccsm4_sie, mask):
    regional_sic = np.nansum(np.nansum((ccsm4_sic*mask),axis=2),axis=1)
    regional_sie = np.nansum(np.nansum((ccsm4_sie*mask),axis=2),axis=1)

    regional_sic_ensmn = np.nanmean(regional_sic,axis=0)
    regional_sie_ensmn = np.nanmean(regional_sie,axis=0)

    regional_sic_high = np.percentile(regional_sic,97.5,axis=0)
    regional_sic_low = np.percentile(regional_sic,2.5,axis=0)
    regional_sie_high = np.percentile(regional_sie,97.5,axis=0)
    regional_sie_low = np.percentile(regional_sie,2.5,axis=0)
    
    return (regional_sie_ensmn,regional_sie_low,regional_sie_high,
            regional_sic_ensmn,regional_sic_low,regional_sic_high)

def load_regional_masks():
    reg_filename = ('/home/disk/p/mkb22/Documents/si_analysis_kb/common_era_experiments/analysis/'+
                    'regionmask_surfaceareacell_2x2_grid.nc')
    reg_mask = xr.open_dataset(reg_filename)
    
    regional_masks = {}

    regional_masks['Sea of Okhotsk'] = reg_mask.cell_area.where(reg_mask.region_mask==2)/(1e6)
    regional_masks['Bering Sea'] = reg_mask.cell_area.where(reg_mask.region_mask==3)/(1e6)
    regional_masks['Hudson Bay'] = reg_mask.cell_area.where(reg_mask.region_mask==4)/(1e6)
    regional_masks['St John'] = reg_mask.cell_area.where(reg_mask.region_mask==5)/(1e6)
    regional_masks['Baffin Bay'] = reg_mask.cell_area.where(reg_mask.region_mask==6)/(1e6)
    regional_masks['East Greenland Sea'] = reg_mask.cell_area.where(reg_mask.region_mask==7)/(1e6)
    regional_masks['Barents Sea'] = reg_mask.cell_area.where(reg_mask.region_mask==8)/(1e6)
    regional_masks['Kara Sea'] = reg_mask.cell_area.where(reg_mask.region_mask==9)/(1e6)
    regional_masks['Laptev Sea'] = reg_mask.cell_area.where(reg_mask.region_mask==10)/(1e6)
    regional_masks['East Siberian Sea'] = reg_mask.cell_area.where(reg_mask.region_mask==11)/(1e6)
    regional_masks['Chukchi Sea'] = reg_mask.cell_area.where(reg_mask.region_mask==12)/(1e6)
    regional_masks['Beaufort Sea'] = reg_mask.cell_area.where(reg_mask.region_mask==13)/(1e6)
    regional_masks['Canadian Archipelago'] = reg_mask.cell_area.where(reg_mask.region_mask==14)/(1e6)
    regional_masks['Central Arctic'] = reg_mask.cell_area.where(reg_mask.region_mask==15)/(1e6)
    
    return regional_masks