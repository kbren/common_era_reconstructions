import sys,os,copy

import sys
import numpy as np
import pickle

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

from time import time
from spharm import Spharmt, getspecindx, regrid
from netCDF4 import Dataset
from scipy import stats

sys.path.insert(1,'/home/disk/kalman2/mkb22/LMR_lite/')
import LMR_lite_utils as LMRlite
import LMR_utils 
import LMR_config

sys.path.insert(1,'/home/disk/p/mkb22/Documents/si_utils_kb/')
import Sice_utils as siutils 

sys.path.insert(1,'/home/disk/p/mkb22/Documents/si_analysis_kb/common_era_experiments/')
import commonera_utils as ceutils 

#--------------------------------------------------------------------
# USER PARAMETERS: 
#--------------------------------------------------------------------
# inflate the sea ice variable here (can inflate whole state here too)
inflate = 2.6
inf_name = '2_6'

#prior_name = 'mpi'
prior_name = 'ccsm4'
obs_name = 'mpi'

pp_err = 0.1
serr = '0_1'

cfile_ccsm4 = '/home/disk/kalman2/mkb22/LMR_lite/configs/config_ccsm4_brennan2020.yml'
cfile_mpi = '/home/disk/kalman2/mkb22/LMR_lite/configs/config_mpi_brennan2020.yml'

proxies = 'pseudo_sister_mpi_truth'

savedir = ('/home/disk/p/mkb22/nobackup/LMR_output/common_era_experiments/experiments/pseudo/')

print('loading configuration...')
cfg_ccsm4 = LMRlite.load_config_simple(cfile_ccsm4)
cfg_mpi = LMRlite.load_config_simple(cfile_mpi)

recon_start = str(cfg_ccsm4.core.recon_period[0])
recon_end = str(cfg_ccsm4.core.recon_period[1])
loc_list = [cfg_ccsm4.core.loc_rad]
    
iter_range = cfg_ccsm4.wrapper.iter_range
MCiters = range(iter_range[0],iter_range[1]+1)

proxy_ind = np.zeros((iter_range[1]+1,405))
prior_ind = np.zeros((iter_range[1]+1,200))

# END USER PARAMETERS: 
#--------------------------------------------------------------------

for iter_num in MCiters:
    savename = ('sic_'+prior_name+'_anrecon_'+str(recon_start)+'_'+
                 str(recon_end)+'_'+proxies+'_inf'+inf_name+'_loc'+str(loc_list[0])+
                '_R'+serr+'_iter'+str(iter_num)+'.pkl')
        
    print('Starting iteration '+str(iter_num)+': inflation = '+str(inflate)+', '+inf_name)

    cfg_dict = LMR_utils.param_cfg_update('core.curr_iter',iter_num)
    
    if cfg_ccsm4.wrapper.multi_seed is not None:
        try:
            curr_seed = cfg_ccsm4.wrapper.multi_seed[iter_num]
            cfg_ccsm4.core.seed = curr_seed 
            cfg_ccsm4.proxies.seed = curr_seed
            cfg_ccsm4.prior.seed = curr_seed
            
            print('Setting current prior iteration seed: {}'.format(curr_seed))
        except IndexError:
            print('ERROR: multi_seed activated but current MC iteration out of'
                  ' range for list of seed values provided in config.')
            raise SystemExit(1)
            
    #--------------------------------------------------------------------       
    print('loading prior')
    X, Xb_one = LMRlite.load_prior(cfg_ccsm4)
    Xbp = Xb_one - Xb_one.mean(axis=1,keepdims=True)
    
    # use this for labeling graphics and output files
    prior_id_string = X.prior_datadir.split('/')[-1]
    print('prior string label: ',prior_id_string)
    
    prior_lat_full = X.prior_dict['tas_sfc_Amon']['lat']
    prior_lon_full = X.prior_dict['tas_sfc_Amon']['lon']

    prior_sic_lat_full = X.prior_dict['sic_sfc_OImon']['lat']
    prior_sic_lon_full = X.prior_dict['sic_sfc_OImon']['lon']
    
    # use this for labeling graphics and output files
    prior_id_string = X.prior_datadir.split('/')[-1]
    print('prior string label: ',prior_id_string)

    # check if config is set to regrid the prior
    if cfg_ccsm4.prior.regrid_method:
        print('regridding prior...')
        # this function over-writes X, even if return is given a different name
        [X_regrid,Xb_one_new] = LMRlite.prior_regrid(cfg_ccsm4,X,Xb_one,verbose=True)
    else:
        X_regrid.trunc_state_info = X.full_state_info
            
    # make a grid object for the prior
    grid = LMRlite.Grid(X_regrid)

    # locate 2m air temperature in Xb_one and make a new array
    tas_pos = X_regrid.trunc_state_info['tas_sfc_Amon']['pos']
    tas = Xb_one_new[tas_pos[0]:tas_pos[1]+1,:]

    # fix using file system softlink for ccsm4 filename...
    sic_pos = X_regrid.trunc_state_info['sic_sfc_OImon']['pos']
    print('assigning '+prior_id_string+ ' sea ice ...')
    sic = Xb_one_new[sic_pos[0]:sic_pos[1]+1,:]
    
    #--------------------------------------------------------------------  
    # Load full prior for tas and regrid to draw observations from 
    print('loading prior for observations')
    X_mpi, Xb_one_mpi = LMRlite.load_prior(cfg_mpi)
    
    prior_mpi_lat_full = X_mpi.prior_dict['tas_sfc_Amon']['lat']
    prior_mpi_lon_full = X_mpi.prior_dict['tas_sfc_Amon']['lon']

    prior_mpi_sic_lat_full = X_mpi.prior_dict['sic_sfc_OImon']['lat']
    prior_mpi_sic_lon_full = X_mpi.prior_dict['sic_sfc_OImon']['lon']
    
    prior_mpi_tas_orig = X_mpi.prior_dict['tas_sfc_Amon']['value']

    temp = np.reshape(prior_mpi_tas_orig,(prior_mpi_tas_orig.shape[0],
                                          prior_mpi_tas_orig.shape[1]*prior_mpi_tas_orig.shape[2]))
    prior_mpi_regrid_tas_prep = np.transpose(temp)
    
    nyears = 1000

    # Regrid full tas prior for proxy selection and verification: 
    [prior_mpi_tas_regrid,
     lat_mpi_tas_new,lon_mpi_tas_new] = LMR_utils.regrid_esmpy(cfg_mpi.prior.esmpy_grid_def['nlat'],
                                                               cfg_mpi.prior.esmpy_grid_def['nlon'],
                                                               nyears,
                                                               prior_mpi_regrid_tas_prep,
                                                               prior_mpi_lat_full,
                                                               prior_mpi_lon_full,
                                                               prior_mpi_lat_full.shape[0],
                                                               prior_mpi_lat_full.shape[1],
                                                               method=cfg_mpi.prior.esmpy_interp_method)

    prior_mpi_tas_regrid = np.reshape(np.transpose(prior_mpi_tas_regrid),
                                      (nyears,cfg_mpi.prior.esmpy_grid_def['nlat'],
                                       cfg_mpi.prior.esmpy_grid_def['nlon'],))
    
    #--------------------------------------------------------------------  
    # inflate the entire state vector
    if 2 == 1:
        print('inflating full state vector...')
        xbm = np.mean(Xb_one_new,1)
        xbp = Xb_one_new - xbm[:,None]
        Xb_one_inflate = np.copy(Xb_one_new)
        Xb_one_inflate = np.add(inflate*xbp,xbm[:,None])
    else:
        # inflate sea ice only
        print('inflating only sea ice by '+str(inflate))
        xb_sicm = np.mean(Xb_one_new[sic_pos[0]:sic_pos[1]+1,:],1)
        xb_sicp = np.squeeze(Xb_one_new[sic_pos[0]:sic_pos[1]+1,:])-xb_sicm[:,None]
        sic_new = np.add(inflate*xb_sicp,xb_sicm[:,None])
        Xb_one_inflate = np.copy(Xb_one_new)
        Xb_one_inflate[sic_pos[0]:sic_pos[1]+1,:] = sic_new
        
    #--------------------------------------------------------------------  
    # load proxies
    prox_manager = LMRlite.load_proxies(cfg_ccsm4)
    
    # NH surface area in M km^2 from concentration in percentage
    nharea = 2*np.pi*(6380**2)/1e8

    #-------------------------------------------------------------------- 
    print('loading Ye...')
    [Ye_assim, 
    Ye_assim_coords] = LMR_utils.load_precalculated_ye_vals_psm_per_proxy(cfg_ccsm4, 
                                                                          prox_manager,
                                                                          'assim',
                                                                          X.prior_sample_indices)
    
    
    #--------------------------------------------------------------------------------
    # Loop over all years available in this reference dataset
    recon_years = range(cfg_ccsm4.core.recon_period[0],cfg_ccsm4.core.recon_period[1])
    #recon_years = range(1979,2000)
    nyears = len(recon_years)

    sic_save = []
    nobs = []
    sic_save_lalo = np.zeros((nyears,grid.nlat,grid.nlon))
    tas_save_lalo = np.zeros((nyears,grid.nlat,grid.nlon))
    sic_lalo_full = np.zeros((grid.nlat,grid.nlon,grid.nens))
    var_save = []
    sic_full_ens = []
    sie_full_ens = []
    sic_ndim = sic_pos[1]-sic_pos[0]+1
    prox_lat = {}
    prox_lon = {}

    begin_time = time()
    yk = -1
    for yk, target_year in enumerate(recon_years):
        print('working on: '+ str(target_year))
        # Get relevant Ye's:
        [vY_real,vR_real,vP_real,vYe,
         vT,vYe_coords] = LMRlite.get_valid_proxies(cfg_ccsm4,prox_manager,target_year,
                                                    Ye_assim,Ye_assim_coords, verbose=False)
        nobs = np.append(nobs,len(vY_real))
        print('Assimilating '+str(len(vY_real))+' proxy locations')

        prox_lat[yk] = vYe_coords[:,0]
        prox_lon[yk] = vYe_coords[:,1]
        vY = []
        vR = []

        for prox in range(len(vY_real)):
            # nearest prior grid lat,lon array indices for the proxy
            tmp = grid.lat[:,0]-np.array(prox_lat[yk])[prox]
            itlat = np.argmin(np.abs(tmp))
            tmp = grid.lon[0,:]-np.array(prox_lon[yk])[prox]
            itlon = np.argmin(np.abs(tmp))

            pseudoproxy = (prior_mpi_tas_regrid[yk,itlat,itlon] +                    
                           np.sqrt(pp_err)*np.random.randn(1))
            vY = np.append(vY,pseudoproxy)
            vR.append(pp_err)

        #xam,Xap,_ = LMRlite.Kalman_optimal(obs_QC,R_QC,Ye_QC,Xb_one_inflate)
        xam,Xap = LMRlite.Kalman_ESRF(cfg_ccsm4,vY,vR,vYe,
                                      Xb_one_inflate,X=X_regrid,
                                      vYe_coords=vYe_coords,verbose=False)

        tas_lalo = np.reshape(xam[tas_pos[0]:tas_pos[1]+1],[grid.nlat,grid.nlon])
        tas_save_lalo[yk,:,:] = tas_lalo

        # this saves sea-ice area for the entire ensemble
        sic_ens = []
        sie_ens = []
     #   for k in range(grid.nens):
        sic_lalo = np.reshape(xam[sic_pos[0]:sic_pos[1]+1,np.newaxis]+Xap[sic_pos[0]:sic_pos[1]+1,:],
                              [grid.nlat,grid.nlon,grid.nens])
        if 'full' in cfg_ccsm4.prior.state_variables['sic_sfc_OImon']:
            sic_lalo = np.where(sic_lalo<0.0,0.0,sic_lalo)
            sic_lalo = np.where(sic_lalo>100.0,100.0,sic_lalo)

            # Calculate extent: 
            sie_lalo = siutils.calc_sea_ice_extent(sic_lalo,15.0)
        else: 
            sic_lalo = sic_lalo
            sie_lalo = np.zeros(sic_lalo.shape)

        for k in range(grid.nens):
            _,nhmic,_ = LMR_utils.global_hemispheric_means(sic_lalo[:,:,k],grid.lat[:, 0])
            _,sie_nhmic,_ = LMR_utils.global_hemispheric_means(sie_lalo[:,:,k],grid.lat[:, 0])
            sic_ens.append(nhmic)
            sie_ens.append(sie_nhmic)

        sic_save_lalo[yk,:,:] = np.nanmean(sic_lalo,axis=2)
        var_save.append(np.var(sic_ens,ddof=1))
        sic_full_ens.append(sic_ens)
        sie_full_ens.append(sie_ens)

        # this saves the gridded concentration field for the entire ensemble
        #Xap_save[0,:,:] = Xap[sic_pos[0]:sic_pos[1]+1]
        #Xap_var = np.var(Xap[sic_pos[0]:sic_pos[1]+1,:],axis=1,ddof=1)
        #Xap_var_save[0,:] = Xap_var

        print('done reconstructing: ',target_year)

    elapsed_time = time() - begin_time
    print('-----------------------------------------------------')
    print('completed in ' + str(elapsed_time) + ' seconds')
    print('-----------------------------------------------------')
    
    sic_recon = {}
    sic_recon['sic_lalo'] = sic_save_lalo
    sic_recon['tas_lalo'] = tas_save_lalo
    sic_recon['sic_ens_var'] = np.mean(var_save)
    sic_recon['nobs'] = nobs
    sic_recon['sia_ens'] = np.squeeze(np.array(sic_full_ens))*nharea
    sic_recon['sie_ens'] = np.squeeze(np.array(sie_full_ens))*nharea
    sic_recon['recon_years'] = recon_years
    sic_recon['Ye_assim'] = Ye_assim
    sic_recon['Ye_assim_coords'] = Ye_assim_coords
    sic_recon['Xb_inflate'] = Xb_one_inflate

    #sic_recon['sic_ens_full'] = sic_ens_full
    #sic_recon['sic_full_ens'] = sic_full_ens
    #sic_recon['obs_full'] = obs_full

    #-----------------------------------------------------
    # SAVE OUTPUT: 
    print('Saving experiment to: ',savedir+savename)
    pickle.dump(sic_recon,open(savedir+savename, "wb"))
    #-----------------------------------------------------
    