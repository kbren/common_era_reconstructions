import sys
import numpy as np
import pickle

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

sys.path.insert(1,'/home/disk/kalman2/mkb22/LMR_lite/')
import LMR_utils 

sys.path.insert(1,'/home/disk/p/mkb22/Documents/si_utils_kb/')
import Sice_utils as siutils 

def sub_arctic_plot(fig,ax,VAR1,LAT,LON,TITLE1,MAX1,colorbar=True,extent=True):
    var1, lon1 = add_cyclic_point(VAR1, coord=LON)
    new_lon2d, new_lat2d = np.meshgrid(lon1, LAT)
    if extent is True: 
        ax.set_extent([-150, 140, 50, 90], crs=ccrs.PlateCarree())
    ax.gridlines(linestyle='--')
    ax.add_feature(cfeature.LAND, facecolor=(1, 1, 1))
    cs = ax.pcolormesh(new_lon2d, new_lat2d, var1, 
                       vmin=-MAX1, vmax=MAX1, cmap=plt.cm.RdBu_r, 
                       transform=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=0.5)
    if colorbar is True:
        plt.colorbar(cs, ax=ax)
    ax.set_title(TITLE1)

def load_recon(LOC, prox_loc=True):
    recon = pickle.load(open(LOC,'rb'))
    
    sic_lalo = recon['sic_lalo']
    tas_lalo = recon['tas_lalo']
    sic_ens_var = recon['sic_ens_var']
    nobs = recon['nobs']
    sia_ens = recon['sia_ens']
    sie_ens = recon['sie_ens']
    recon_years = recon['recon_years']
    Ye_assim = recon['Ye_assim']
    Ye_assim_coords = recon['Ye_assim_coords']
    Xb_inflate = recon['Xb_inflate'] 
    
    if prox_loc is True: 
        proxy_assim_loc = recon['proxy_assim_loc']
    
        return (sic_lalo, tas_lalo, sic_ens_var, nobs, sia_ens, sie_ens, recon_years, 
                Ye_assim, Ye_assim_coords, Xb_inflate, proxy_assim_loc)
    else: 
        return (sic_lalo, tas_lalo, sic_ens_var, nobs, sia_ens, sie_ens, recon_years, 
                Ye_assim, Ye_assim_coords, Xb_inflate)
    
def load_recon_pseudo(loc, sit=False, gmt=False): 
    recon = pickle.load(open(loc,'rb'))

    sic_lalo = recon['sic_lalo']
    tas_lalo = recon['tas_lalo']
    sic_ens_var = recon['sic_ens_var']
    nobs = recon['nobs']
    obs_loc = recon['obs_loc']
    sia_Nens = recon['sia_Nens']
    sie_Nens = recon['sie_Nens']
    sia_Sens = recon['sia_Sens']
    sie_Sens = recon['sie_Sens']
    recon_years = recon['recon_years']
    Xb_inflate = recon['Xb_inflate']
    prox_lat = recon['prox_lat']
    prox_lon = recon['prox_lon']
    tas_truth = recon['tas_truth']
    
    if sit is True: 
        sit_lalo = recon['sit_lalo']
        sit_Nens = recon['sit_Nens']
        sit_Sens = recon['sit_Sens']
        
        if gmt is True: 
            tas_Gens = recon['tas_Gens']
            tas_Nens = recon['tas_Nens']
        
            return [sic_lalo, tas_lalo, sic_ens_var, 
                    nobs, obs_loc, sia_Nens, sie_Nens, 
                    sia_Sens, sie_Sens, recon_years, Xb_inflate,
                    prox_lat, prox_lon, tas_truth, 
                    sit_lalo, sit_Nens, sit_Sens, tas_Gens, tas_Nens]
        else: 
            return [sic_lalo, tas_lalo, sic_ens_var, 
                    nobs, obs_loc, sia_Nens, sie_Nens, 
                    sia_Sens, sie_Sens, recon_years, Xb_inflate,
                    prox_lat, prox_lon, tas_truth, 
                    sit_lalo, sit_Nens, sit_Sens]
    
    else: 
        if gmt is True: 
            tas_Gens = recon['tas_Gens']
            tas_Nens = recon['tas_Nens']
        
            return [sic_lalo, tas_lalo, sic_ens_var, 
                    nobs, obs_loc, sia_Nens, sie_Nens, 
                    sia_Sens, sie_Sens, recon_years, Xb_inflate,
                    prox_lat, prox_lon, tas_truth, 
                    sit_lalo, sit_Nens, sit_Sens, tas_Gens, tas_Nens]
        else: 
            return [sic_lalo, tas_lalo, sic_ens_var, 
                    nobs, obs_loc, sia_Nens, sie_Nens, 
                    sia_Sens, sie_Sens, recon_years, Xb_inflate,
                    prox_lat, prox_lon, tas_truth]

def load_recon_allit(output_dir,filename,niter,prox_loc=True):

    for it in range(niter):
        output_file = filename[:-5]+str(it)+'.pkl'

        if prox_loc is True: 
            [sic_lalo, tas_lalo, sic_ens_var, 
             nobs, sia_ens, sie_ens, recon_years, 
             Ye_assim, Ye_assim_coords, Xb_inflate, proxy_assim_loc] = load_recon(output_dir+output_file)          
        else: 
            [sic_lalo, tas_lalo, sic_ens_var, 
             nobs, sia_ens, sie_ens, recon_years, 
             Ye_assim, Ye_assim_coords, Xb_inflate] = load_recon(output_dir+output_file, prox_loc=False)

        if it is 0: 
            sic_lalo_allit = np.zeros((sic_lalo.shape[0],sic_lalo.shape[1],sic_lalo.shape[2],niter))
            tas_lalo_allit = np.zeros((tas_lalo.shape[0],tas_lalo.shape[1],tas_lalo.shape[2],niter))
            sic_ens_var_allit = np.zeros(niter)
            nobs_allit = np.zeros((nobs.shape[0],niter))
            sia_ens_allit = np.zeros((sia_ens.shape[0],sia_ens.shape[1],niter))
            sie_ens_allit = np.zeros((sie_ens.shape[0],sie_ens.shape[1],niter))
            Ye_assim_allit = np.zeros((Ye_assim.shape[0],Ye_assim.shape[1],niter))
            Ye_assim_coords_allit = np.zeros((Ye_assim_coords.shape[0],Ye_assim_coords.shape[1],niter))
            Xb_inflate_allit = np.zeros((Xb_inflate.shape[0],Xb_inflate.shape[1],niter))
            if prox_loc is True: 
                proxy_assim_loc_allit = {}

        sic_lalo_allit[:,:,:,it] = sic_lalo
        tas_lalo_allit[:,:,:,it] = tas_lalo
        sic_ens_var_allit[it] = niter
        nobs_allit[:,it] = nobs
        sia_ens_allit[:,:,it] = sia_ens
        sie_ens_allit[:,:,it] = sie_ens
        Ye_assim_allit[:,:,it] = Ye_assim
        Ye_assim_coords_allit[:,:,it] = Ye_assim_coords
        Xb_inflate_allit[:,:,it] = Xb_inflate
        if prox_loc is True: 
            proxy_assim_loc_allit['iter '+str(it)] = proxy_assim_loc
        
    if prox_loc is True:        
        return (sic_lalo_allit, tas_lalo_allit, sic_ens_var_allit, nobs_allit, 
                sia_ens_allit, sie_ens_allit, Ye_assim_allit, Ye_assim_coords_allit, 
                Xb_inflate_allit,recon_years, proxy_assim_loc_allit)
    else: 
        return (sic_lalo_allit, tas_lalo_allit, sic_ens_var_allit, nobs_allit, 
                sia_ens_allit, sie_ens_allit, Ye_assim_allit, Ye_assim_coords_allit, 
                Xb_inflate_allit,recon_years)
    
def load_pseudo_recon_allit(output_dir,filename,niter):

    for it in range(niter):
        output_file = filename[:-5]+str(it)+'.pkl'

        [sic_lalo, tas_lalo, sic_ens_var, 
        nobs, obs_loc, sia_Nens, sie_Nens, 
        sia_Sens, sie_Sens, recon_years, Xb_inflate,
        prox_lat, prox_lon, tas_truth] = load_recon_pseudo(output_dir+output_file)

        if it is 0: 
            sic_lalo_allit = np.zeros((sic_lalo.shape[0],sic_lalo.shape[1],sic_lalo.shape[2],niter))
            tas_lalo_allit = np.zeros((tas_lalo.shape[0],tas_lalo.shape[1],tas_lalo.shape[2],niter))
            sic_ens_var_allit = np.zeros(niter)
            nobs_allit = np.zeros((nobs.shape[0],niter))
            obs_loc_allit = {}
            sia_Nens_allit = np.zeros((sia_Nens.shape[0],sia_Nens.shape[1],niter))
            sie_Nens_allit = np.zeros((sie_Nens.shape[0],sie_Nens.shape[1],niter))
            sia_Sens_allit = np.zeros((sia_Sens.shape[0],sia_Sens.shape[1],niter))
            sie_Sens_allit = np.zeros((sie_Sens.shape[0],sie_Sens.shape[1],niter))
            Xb_inflate_allit = np.zeros((Xb_inflate.shape[0],Xb_inflate.shape[1],niter))
            prox_lat = {}
            prox_lon = {}
            tas_truth_allit = np.zeros((tas_truth.shape[0],tas_truth.shape[1],
                                        tas_truth.shape[2],niter))

        sic_lalo_allit[:,:,:,it] = sic_lalo
        tas_lalo_allit[:,:,:,it] = tas_lalo
        sic_ens_var_allit[it] = sic_ens_var
        nobs_allit[:,it] = nobs
        obs_loc_allit['iter '+str(it)] = obs_loc
        sia_Nens_allit[:,:,it] = sia_Nens
        sie_Nens_allit[:,:,it] = sie_Nens
        sia_Sens_allit[:,:,it] = sia_Sens
        sie_Sens_allit[:,:,it] = sie_Sens
        Xb_inflate_allit[:,:,it] = Xb_inflate
        prox_lat['iter '+str(it)] = prox_lat
        prox_lon['iter '+str(it)] = prox_lon
        tas_truth_allit[:,:,:,it] = tas_truth
        
    return [sic_lalo_allit, tas_lalo_allit, sic_ens_var_allit, nobs_allit, 
            obs_loc_allit, sia_Nens_allit, sie_Nens_allit, sia_Sens_allit, 
            sie_Sens_allit, Xb_inflate_allit, prox_lat, prox_lon, 
            recon_years, tas_truth_allit]
 
def load_sit_pseudo_recon_allit(output_dir,filename,niter):

    for it in range(niter):
        output_file = filename[:-5]+str(it)+'.pkl'

        [sic_lalo, tas_lalo, sic_ens_var, 
        nobs, obs_loc, sia_Nens, sie_Nens, 
        sia_Sens, sie_Sens, recon_years, Xb_inflate,
        prox_lat, prox_lon, tas_truth,
        sit_lalo, sit_Nens, sit_Sens] = load_recon_pseudo(output_dir+output_file, sit=True)

        if it is 0: 
            sic_lalo_allit = np.zeros((sic_lalo.shape[0],sic_lalo.shape[1],sic_lalo.shape[2],niter))
            tas_lalo_allit = np.zeros((tas_lalo.shape[0],tas_lalo.shape[1],tas_lalo.shape[2],niter))
            sic_ens_var_allit = np.zeros(niter)
            nobs_allit = np.zeros((nobs.shape[0],niter))
            obs_loc_allit = {}
            sia_Nens_allit = np.zeros((sia_Nens.shape[0],sia_Nens.shape[1],niter))
            sie_Nens_allit = np.zeros((sie_Nens.shape[0],sie_Nens.shape[1],niter))
            sia_Sens_allit = np.zeros((sia_Sens.shape[0],sia_Sens.shape[1],niter))
            sie_Sens_allit = np.zeros((sie_Sens.shape[0],sie_Sens.shape[1],niter))
            Xb_inflate_allit = np.zeros((Xb_inflate.shape[0],Xb_inflate.shape[1],niter))
            prox_lat = {}
            prox_lon = {}
            tas_truth_allit = np.zeros((tas_truth.shape[0],tas_truth.shape[1],
                                        tas_truth.shape[2],niter))
            sit_lalo_allit = np.zeros((sit_lalo.shape[0],sit_lalo.shape[1],sit_lalo.shape[2],niter))
            sit_Nens_allit = np.zeros((sit_Nens.shape[0],sit_Nens.shape[1],niter))
            sit_Sens_allit = np.zeros((sit_Sens.shape[0],sit_Sens.shape[1],niter))

        sic_lalo_allit[:,:,:,it] = sic_lalo
        tas_lalo_allit[:,:,:,it] = tas_lalo
        sic_ens_var_allit[it] = sic_ens_var
        nobs_allit[:,it] = nobs
        obs_loc_allit['iter '+str(it)] = obs_loc
        sia_Nens_allit[:,:,it] = sia_Nens
        sie_Nens_allit[:,:,it] = sie_Nens
        sia_Sens_allit[:,:,it] = sia_Sens
        sie_Sens_allit[:,:,it] = sie_Sens
        Xb_inflate_allit[:,:,it] = Xb_inflate
        prox_lat['iter '+str(it)] = prox_lat
        prox_lon['iter '+str(it)] = prox_lon
        tas_truth_allit[:,:,:,it] = tas_truth
        sit_lalo_allit[:,:,:,it] = sit_lalo
        sit_Nens_allit[:,:,it] = sit_Nens
        sit_Sens_allit[:,:,it] = sit_Sens

    return [sic_lalo_allit, tas_lalo_allit, sic_ens_var_allit, nobs_allit, 
            obs_loc_allit, sia_Nens_allit, sie_Nens_allit, sia_Sens_allit, 
            sie_Sens_allit, Xb_inflate_allit, prox_lat, prox_lon, 
            recon_years, tas_truth_allit, sit_lalo_allit, 
            sit_Nens_allit, sit_Sens_allit]

def load_sit_gmt_pseudo_recon_allit(output_dir,filename,niter):

    for it in range(niter):
        output_file = filename[:-5]+str(it)+'.pkl'

        [sic_lalo, tas_lalo, sic_ens_var, 
        nobs, obs_loc, sia_Nens, sie_Nens, 
        sia_Sens, sie_Sens, recon_years, Xb_inflate,
        prox_lat, prox_lon, tas_truth,
        sit_lalo, sit_Nens, sit_Sens,
        tas_Gens, tas_Nens] = load_recon_pseudo(output_dir+filename, sit=True, gmt=True)

        if it is 0: 
            sic_lalo_allit = np.zeros((sic_lalo.shape[0],sic_lalo.shape[1],sic_lalo.shape[2],niter))
            tas_lalo_allit = np.zeros((tas_lalo.shape[0],tas_lalo.shape[1],tas_lalo.shape[2],niter))
            sic_ens_var_allit = np.zeros(niter)
            nobs_allit = np.zeros((nobs.shape[0],niter))
            obs_loc_allit = {}
            sia_Nens_allit = np.zeros((sia_Nens.shape[0],sia_Nens.shape[1],niter))
            sie_Nens_allit = np.zeros((sie_Nens.shape[0],sie_Nens.shape[1],niter))
            sia_Sens_allit = np.zeros((sia_Sens.shape[0],sia_Sens.shape[1],niter))
            sie_Sens_allit = np.zeros((sie_Sens.shape[0],sie_Sens.shape[1],niter))
            Xb_inflate_allit = np.zeros((Xb_inflate.shape[0],Xb_inflate.shape[1],niter))
            prox_lat = {}
            prox_lon = {}
            tas_truth_allit = np.zeros((tas_truth.shape[0],tas_truth.shape[1],
                                        tas_truth.shape[2],niter))
            sit_lalo_allit = np.zeros((sit_lalo.shape[0],sit_lalo.shape[1],sit_lalo.shape[2],niter))
            sit_Nens_allit = np.zeros((sit_Nens.shape[0],sit_Nens.shape[1],niter))
            sit_Sens_allit = np.zeros((sit_Sens.shape[0],sit_Sens.shape[1],niter))
            tas_Nens_allit = np.zeros((tas_Nens.shape[0],tas_Nens.shape[1],niter))
            tas_Gens_allit = np.zeros((tas_Gens.shape[0],tas_Gens.shape[1],niter))

        sic_lalo_allit[:,:,:,it] = sic_lalo
        tas_lalo_allit[:,:,:,it] = tas_lalo
        sic_ens_var_allit[it] = sic_ens_var
        nobs_allit[:,it] = nobs
        obs_loc_allit['iter '+str(it)] = obs_loc
        sia_Nens_allit[:,:,it] = sia_Nens
        sie_Nens_allit[:,:,it] = sie_Nens
        sia_Sens_allit[:,:,it] = sia_Sens
        sie_Sens_allit[:,:,it] = sie_Sens
        Xb_inflate_allit[:,:,it] = Xb_inflate
        prox_lat['iter '+str(it)] = prox_lat
        prox_lon['iter '+str(it)] = prox_lon
        tas_truth_allit[:,:,:,it] = tas_truth
        sit_lalo_allit[:,:,:,it] = sit_lalo
        sit_Nens_allit[:,:,it] = sit_Nens
        sit_Sens_allit[:,:,it] = sit_Sens
        tas_Nens_allit[:,:,it] = tas_Nens
        tas_Gens_allit[:,:,it] = tas_Gens

    return [sic_lalo_allit, tas_lalo_allit, sic_ens_var_allit, nobs_allit, 
            obs_loc_allit, sia_Nens_allit, sie_Nens_allit, sia_Sens_allit, 
            sie_Sens_allit, Xb_inflate_allit, prox_lat, prox_lon, 
            recon_years, tas_truth_allit, sit_lalo_allit, 
            sit_Nens_allit, sit_Sens_allit, tas_Gens_allit, tas_Nens_allit]

    
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
    fet_sia_adj = fet_data['si_area_adj']
    fet_time = fet_data['time'][1:]

    # Find anomalies: 

    # Calculate mean 
    fet_anom_cent_sia = np.nanmean(fet_sia[np.where(fet_time<=ANOM_END)],axis=0)
    fet_anom_cent_sia_adj = np.nanmean(fet_sia_adj[np.where(fet_time[1:]<=ANOM_END)],axis=0)
    fet_anom_cent_sie = np.nanmean(fet_sie[np.where(fet_time<=ANOM_END)],axis=0)

    # Find anomalies:  
    fet_sia_anom = fet_sia - fet_anom_cent_sia
    fet_sia_anom_adj = fet_sia_adj - fet_anom_cent_sia_adj
    fet_sie_anom = fet_sie - fet_anom_cent_sie
    
    return fet_sia_anom, fet_sia_anom_adj, fet_sie_anom, fet_time

def load_annual_walsh(version=None):
    """
    Loads annualized total Arctic extent and area from Walsh et al version 1 or 2. 
    
    INPUTS: 
    version: integer value indicating the value of Walsh data to load (takes values: 1,2,None)
    
    OUTPUTS: 
    walsh_sia: total arctic sea ice area (km^2)
    walsh_sie: total arctic sea ice extent (km^2)
    walsh_time: years of walsh data (years)
    """
    walsh_directory = '/home/disk/chaos/mkb22/Documents/SeaIceData/Walsh2016/walsh_comparison/'
    
    if version is None: 
        vera = 2
        vere = 2
    elif version == 3: 
        vera = 2
        vere = 3
    else: 
        vera = version
        vere = version

    walsh_sie_file_an = 'Walsh_annual_sie_km2_v'+str(vere)+'.npz'
    walsh_sia_file_an = 'Walsh_annual_sia_km2_v'+str(vera)+'.npz'

    walsh_sie_data_an = np.load(walsh_directory + walsh_sie_file_an)
    walsh_sia_data_an = np.load(walsh_directory + walsh_sia_file_an)
    
    walsh_time = walsh_sie_data_an['years']
    
    if version ==2: 
        walsh_sia = walsh_sia_data_an['walsh_nh_annual_area_total']/1e6
        walsh_sie = walsh_sie_data_an['walsh_nh_annual_extent_total']/1e6
    elif version ==3:     
        walsh_sia = walsh_sia_data_an['walsh_nh_annual_area_total']/1e6
        walsh_sie = walsh_sie_data_an['walsh_nh_annual_extent_total_amn_first']/1e6
    else: 
        walsh_sia = walsh_sia_data_an['walsh_nh_annual_area_total']
        walsh_sie = walsh_sie_data_an['walsh_nh_annual_extent_total']
    
#     if ver == 1: 
#         walsh_time = np.arange(1850,2014,1)
#     else: 
#         walsh_time = np.arange(1850,2018,1)
    
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

def draw_pseudoproxy(grid_pseudo, ref_ob_lat, ref_ob_lon, prior_anom, r): 
    tmp = grid_pseudo.lat[:,0]-ref_ob_lat
    itlat = np.argmin(np.abs(tmp))
    tmp = grid_pseudo.lon[0,:]-ref_ob_lon
    itlon = np.argmin(np.abs(tmp))

    # draw the proxy and prior estimates
    pseudoproxy = (prior_anom[itlat,itlon]+np.sqrt(r)*np.random.randn(1))
    
    return pseudoproxy, itlat, itlon

def load_mpi_lm_regridded():
    mpi_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/MPI/'
    mpi_file = 'mpi_sic_tas_20CRv2_850_1850_full.npz'

    mpi_lm = np.load(mpi_dir+mpi_file)

    mpi_truth_sic = mpi_lm['sic']
    mpi_truth_tas = mpi_lm['tas']
    mpi_truth_lat = mpi_lm['lat']
    mpi_truth_lon = mpi_lm['lon']

    return mpi_truth_tas, mpi_truth_sic, mpi_truth_lat, mpi_truth_lon

def load_ccsm4_lm_regridded():
    ccsm4_dir = '/home/disk/chaos/mkb22/Documents/SeaIceData/CCSM4/CCSM4_last_millennium/'
    ccsm4_file = 'ccsm4_sic_tas_20CRv2_850_1850_full.npz'

    ccsm4_lm = np.load(ccsm4_dir+ccsm4_file)

    ccsm4_truth_sic = ccsm4_lm['sic_ccsm4']
    ccsm4_truth_tas = ccsm4_lm['tas_ccsm4']
    ccsm4_truth_lat = ccsm4_lm['lat_ccsm4']
    ccsm4_truth_lon = ccsm4_lm['lon_ccsm4']
    
    return ccsm4_truth_tas, ccsm4_truth_sic, ccsm4_truth_lat, ccsm4_truth_lon

def calc_lm_tot_si(truth_sic,truth_lat,truth_time,anom_start, anom_end):
    # NH surface area in M km^2 from concentration in percentage
    nharea = 2*np.pi*(6380**2)/1e8

    sie_lalo= siutils.calc_sea_ice_extent(truth_sic,15.0)

    _,nh_sic_truth,sh_sic_truth = LMR_utils.global_hemispheric_means(truth_sic,truth_lat[:,0])
    _,nh_sie_truth,sh_sie_truth= LMR_utils.global_hemispheric_means(sie_lalo,truth_lat[:,0])
    sia_nh_truth = nh_sic_truth*nharea
    sie_nh_truth = nh_sie_truth*nharea
    sia_sh_truth = sh_sic_truth*nharea
    sie_sh_truth = sh_sie_truth*nharea
    
    anom_int = np.where((truth_time>=anom_start)&(truth_time<=anom_end+1))

    sia_nh_truth_anom = sia_nh_truth - np.nanmean(sia_nh_truth[anom_int])
    sie_nh_truth_anom = sie_nh_truth - np.nanmean(sie_nh_truth[anom_int])
    sia_sh_truth_anom = sia_sh_truth - np.nanmean(sia_sh_truth[anom_int])
    sie_sh_truth_anom = sie_sh_truth - np.nanmean(sie_sh_truth[anom_int])
    
    return [sia_nh_truth, sie_nh_truth, sia_nh_truth_anom, sie_nh_truth_anom,
            sia_sh_truth, sie_sh_truth, sia_sh_truth_anom, sie_sh_truth_anom]

def calc_lm_gmt(truth_tas,truth_lat,truth_time,anom_start, anom_end):
    gmt_truth,nht_truth,sht_truth = LMR_utils.global_hemispheric_means(truth_tas,truth_lat[:,0])
    
    anom_int = np.where((truth_time>=anom_start)&(truth_time<=anom_end+1))

    gmt_truth_anom = gmt_truth - np.nanmean(gmt_truth[anom_int])
    nht_truth_anom = nht_truth - np.nanmean(nht_truth[anom_int])
    sht_truth_anom = sht_truth - np.nanmean(sht_truth[anom_int])
    
    return [gmt_truth,nht_truth,sht_truth,
            gmt_truth_anom,nht_truth_anom,sht_truth_anom]

def load_recon_grid(): 
    loc = '/home/disk/p/mkb22/nobackup/LMR_output/reanalysis_reconstruction_data/'
    grid = pickle.load(open(loc +'sic_recon_grid.pkl','rb'))
    
    recon_lat = grid.lat[:,0]
    recon_lon = grid.lon[0,:]
    
    return recon_lat, recon_lon

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
    print('Hello')
    yr_range_var = np.where((VAR_TIME>=START_TIME)&(VAR_TIME<END_TIME+1))
    yr_range_ref = np.where((REF_TIME>=START_TIME)&(REF_TIME<END_TIME+1))
    print(yr_range_var)
    
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