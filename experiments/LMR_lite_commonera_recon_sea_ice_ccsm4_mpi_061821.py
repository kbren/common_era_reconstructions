import sys,os,copy
#sys.path.append("/Users/hakim/gitwork/LMR_python3")

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

sys.path.insert(1,'/home/disk/p/mkb22/Documents/si_analysis_kb/instrumental_assimilation_experiments/')
import reanalysis_recons_utils as rrutils

sys.path.insert(1,'/home/disk/kalman2/mkb22/LMR_lite/')
import LMR_lite_utils as LMRlite
import LMR_utils 
import LMR_config
#import LMR_config_greg

sys.path.insert(1,'/home/disk/p/mkb22/Documents/si_utils_kb/')
import Sice_utils as siutils 
                                          
#--------------------------------------------------------------------
# USER PARAMETERS: 
#--------------------------------------------------------------------
# inflate the sea ice variable here (can inflate whole state here too)

inflate = 2.6
inf_name = '2_6'
#inflate = 1.8
#inf_name = '1_8'

#prior_name = 'mpi'
prior_name = 'ccsm4'

cfile = '/home/disk/kalman2/mkb22/LMR_lite/configs/config_ccsm4_brennan2020.yml'
#cfile = '/home/disk/kalman2/mkb22/LMR_lite/configs/config_mpi_brennan2020.yml'
#cfile = '/home/disk/kalman2/mkb22/LMR_lite/configs/config_ccsm4_fixedprox.yml'
#cfile = '/home/disk/kalman2/mkb22/LMR_lite/configs/config_production2.yml'

proxies = 'pages2kv2'
#proxies = 'fullLMRdbv0_4'
#proxies = 'fixedprox'

savedir = ('/home/disk/p/mkb22/nobackup/LMR_output/common_era_experiments/experiments/ccsm4/')
#savedir = ('/home/disk/p/mkb22/nobackup/LMR_output/common_era_experiments/experiments/mpi/')

print('loading configuration...')
#cfg = None
cfg = LMRlite.load_config(cfile)

recon_start = str(cfg.core.recon_period[0])
recon_end = str(cfg.core.recon_period[1])
loc_list = [cfg.core.loc_rad]
print('Reconstructing years: '+recon_start+'-'+recon_end)
    
#iter_range = cfg.wrapper.iter_range
iter_range = [0,1]
MCiters = range(iter_range[0],iter_range[1]+1)

proxy_ind = np.zeros((iter_range[1]+1,405))
prior_ind = np.zeros((iter_range[1]+1,200))
        
# END USER PARAMETERS: 
#--------------------------------------------------------------------

for iter_num in MCiters:
    savename = ('sic_'+prior_name+'_anrecon_revisions2_testcount'+str(recon_start)+'_'+
                     str(recon_end)+'_'+proxies+'_inf'+inf_name+'_loc'+str(loc_list[0])+
                     '_iter'+str(iter_num)+'.pkl')
        
    print('Starting iteration '+str(iter_num)+': inflation = '+str(inflate)+', '+inf_name)

    cfg_dict = LMR_utils.param_cfg_update('core.curr_iter',iter_num)
    
    if cfg.wrapper.multi_seed is not None:
        try:
            curr_seed = cfg.wrapper.multi_seed[iter_num]
            cfg.core.seed = curr_seed 
            cfg.proxies.seed = curr_seed
            cfg.prior.seed = curr_seed
            
            print('Setting current prior iteration seed: {}'.format(curr_seed))
        except IndexError:
            print('ERROR: multi_seed activated but current MC iteration out of'
                  ' range for list of seed values provided in config.')
            raise SystemExit(1)
            
    #--------------------------------------------------------------------
    print('loading prior')
    X, Xb_one = LMRlite.load_prior(cfg)
    Xbp = Xb_one - Xb_one.mean(axis=1,keepdims=True)
    
    X_orig = X 

    # use this for labeling graphics and output files
    prior_id_string = X.prior_datadir.split('/')[-1]
    print('prior string label: ',prior_id_string)

    # check if config is set to regrid the prior
    if cfg.prior.regrid_method:
        print('regridding prior...')
        # this function over-writes X, even if return is given a different name
        [X_regrid,Xb_one_new] = LMRlite.prior_regrid(cfg,X,Xb_one,verbose=True)
    else:
        X_regrid.trunc_state_info = X.full_state_info

    #--------------------------------------------------------------------
    # INFLATION:
    # make a grid object for the prior
    grid = LMRlite.Grid(X_regrid)

    # locate 2m air temperature in Xb_one and make a new array
    tas_pos = X_regrid.trunc_state_info['tas_sfc_Amon']['pos']
    tas = Xb_one_new[tas_pos[0]:tas_pos[1]+1,:]

    # fix using file system softlink for ccsm4 filename...
    sic_pos = X_regrid.trunc_state_info['sic_sfc_OImon']['pos']
    print('assigning '+prior_id_string+ ' sea ice ...')

    sic = Xb_one_new[sic_pos[0]:sic_pos[1]+1,:]

    # inflate the entire state vector
    if 2 == 1:
        print('inflating full state vector...')
        xbm = np.mean(Xb_one_new,1)
        xbp = Xb_one_new - xbm[:,None]
        Xb_one_inflate = np.copy(Xb_one_new)
        Xb_one_inflate = np.add(inflate*xbp,xbm[:,None])
    else:
        # inflate sea ice only
        print('inflating only sea ice...')
        xb_sicm = np.mean(Xb_one_new[sic_pos[0]:sic_pos[1]+1,:],1)
        xb_sicp = np.squeeze(Xb_one_new[sic_pos[0]:sic_pos[1]+1,:])-xb_sicm[:,None]
        sic_new = np.add(inflate*xb_sicp,xb_sicm[:,None])
        Xb_one_inflate = np.copy(Xb_one_new)
        Xb_one_inflate[sic_pos[0]:sic_pos[1]+1,:] = sic_new

    #--------------------------------------------------------------------
    # load proxies
    prox_manager = LMRlite.load_proxies(cfg)

    print('loading Ye...')
    [Ye_assim, 
    Ye_assim_coords] = LMR_utils.load_precalculated_ye_vals_psm_per_proxy(cfg, 
                                                                          prox_manager,
                                                                          'assim',
                                                                          X.prior_sample_indices)

    #Gather proxy info to save: 
    prox_present = []
    nprox_assim = 0

    for Y in prox_manager.sites_assim_proxy_objs():
        if Y.type not in prox_present:
            prox_present = np.append(prox_present,Y.type)
        nprox_assim = nprox_assim+1

    prox_assim_info = {}

    for P in prox_present: 
        prox_assim_info[P] = {'lat':[],'lon':[]}

    for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
        prox_assim_info[Y.type]['lat'] = np.append(prox_assim_info[Y.type]['lat'],Y.lat)
        prox_assim_info[Y.type]['lon'] = np.append(prox_assim_info[Y.type]['lon'],Y.lon)

    #--------------------------------------------------------------------------------    
    # NH surface area in M km^2 from concentration in percentage
    nharea = 2*np.pi*(6380**2)/1e8

    #--------------------------------------------------------------------------------
    #START DA: 
    #--------------------------------------------------------------------------------
    #cfg.core.loc_rad = loc
    print('Localization radius: ',cfg.core.loc_rad)

    #--------------------------------------------------------------------------------
    # Loop over all years available in this reference dataset
    recon_years = range(cfg.core.recon_period[0],cfg.core.recon_period[1])
    #recon_years = range(1979,2000)
    nyears = len(recon_years)

    sic_save = []
    nobs = []
    sic_save_lalo_og = np.zeros((nyears,grid.nlat,grid.nlon,grid.nens))
    sic_save_lalo = np.zeros((nyears,grid.nlat,grid.nlon))
    tas_save_lalo = np.zeros((nyears,grid.nlat,grid.nlon))
    sic_lalo_full = []
    var_save = []
    sic_full_ens = []
    sie_full_ens = []
    gmtas_full_ens = []
    nhmtas_full_ens = []
    amtas_full_ens = []
    sic_ndim = sic_pos[1]-sic_pos[0]+1
    
    cutoff_count_pos = 0
    cutoff_count_neg = 0

    begin_time = time()
    yk = -1
    for yk, target_year in enumerate(recon_years):
        print('working on: '+ str(target_year))

        # Do data assimilation
        [vY,vR,vP,vYe,
         vT,vYe_coords] = LMRlite.get_valid_proxies(cfg,prox_manager,target_year,
                                                    Ye_assim,Ye_assim_coords, verbose=False)
        nobs = np.append(nobs,len(vY))
        print('Number of obs assimilated = '+str(len(vY)))

        #xam,Xap,_ = LMRlite.Kalman_optimal(obs_QC,R_QC,Ye_QC,Xb_one_inflate)
        xam,Xap = LMRlite.Kalman_ESRF(cfg,vY,vR,vYe,
                                      Xb_one_inflate,X=X_regrid,
                                      vYe_coords=vYe_coords,verbose=False)

        tas_lalo = np.reshape(xam[tas_pos[0]:tas_pos[1]+1],[grid.nlat,grid.nlon])
        tas_save_lalo[yk,:,:] = tas_lalo
        
        tas_lalo_ens = np.reshape(xam[tas_pos[0]:tas_pos[1]+1,np.newaxis]+Xap[sic_pos[0]:sic_pos[1]+1,:],
                                  [grid.nlat,grid.nlon,grid.nens])

        # this loop applies a percent cutoff and 
        # calculates total Arctic  sea-ice area and extent for each ensemble member
        sic_ens = []
        sie_ens = []
        gmtas_ens = []
        nhmtas_ens = []
        amtas_ens = []
        sic_lalo_og = np.reshape(xam[sic_pos[0]:sic_pos[1]+1,np.newaxis]+Xap[sic_pos[0]:sic_pos[1]+1,:],
                                 [grid.nlat,grid.nlon,grid.nens])
        if 'full' in cfg.prior.state_variables['sic_sfc_OImon']:
            cutoff_count_pos = cutoff_count_pos + (sic_lalo_og[grid.lat[:,0]>0,:,:]>100.0).sum()
            cutoff_count_neg = cutoff_count_neg + (sic_lalo_og[grid.lat[:,0]>0,:,:]<0.0).sum()
            
            sic_lalo = np.where(sic_lalo_og<0.0,0.0,sic_lalo_og)
            sic_lalo = np.where(sic_lalo>100.0,100.0,sic_lalo)

            # Calculate extent: 
            sie_lalo = siutils.calc_sea_ice_extent(sic_lalo,15.0)
        else: 
            sic_lalo = sic_lalo
            sie_lalo = np.zeros(sic_lalo.shape)

        for k in range(grid.nens):
            _,nhmic,_ = LMR_utils.global_hemispheric_means(sic_lalo[:,:,k],grid.lat[:, 0])
            _,sie_nhmic,_ = LMR_utils.global_hemispheric_means(sie_lalo[:,:,k],grid.lat[:, 0])
            gmtas,nhmtas,_ = LMR_utils.global_hemispheric_means(tas_lalo_ens[:,:,k],grid.lat[:, 0])
            amtas,_,_ = LMR_utils.global_hemispheric_means(tas_lalo_ens[grid.lat[:,0]>=60.0,:,k],
                                                           grid.lat[(grid.lat[:,0]>=60.0),0])
            
            sic_ens.append(nhmic)
            sie_ens.append(sie_nhmic)
            gmtas_ens.append(gmtas)
            nhmtas_ens.append(nhmtas)
            amtas_ens.append(amtas)
        
        sic_save_lalo[yk,:,:] = np.nanmean(sic_lalo,axis=2)
        var_save.append(np.var(sic_ens,ddof=1))
        sic_full_ens.append(sic_ens)
        sie_full_ens.append(sie_ens)
        gmtas_full_ens.append(gmtas_ens)
        nhmtas_full_ens.append(nhmtas_ens)
        amtas_full_ens.append(amtas_ens)

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
    sic_recon['gmtas_ens'] = np.squeeze(np.array(gmtas_full_ens))
    sic_recon['nhmtas_ens'] = np.squeeze(np.array(nhmtas_full_ens))
    sic_recon['amtas_ens'] = np.squeeze(np.array(amtas_full_ens))
    sic_recon['recon_years'] = recon_years
    sic_recon['Ye_assim'] = Ye_assim
    sic_recon['Ye_assim_coords'] = Ye_assim_coords
    sic_recon['Xb_inflate'] = Xb_one_inflate
    sic_recon['proxy_assim_loc'] = prox_assim_info
    sic_recon['cutoff_count_pos'] = cutoff_count_pos
    sic_recon['cutoff_count_neg'] = cutoff_count_neg

    #sic_recon['sic_ens_full'] = sic_ens_full
    #sic_recon['sic_full_ens'] = sic_full_ens
    #sic_recon['obs_full'] = obs_full

    #-----------------------------------------------------
    # SAVE OUTPUT: 
    print('Saving experiment to: ',savedir+savename)
    pickle.dump(sic_recon,open(savedir+savename, "wb"))
    #-----------------------------------------------------


