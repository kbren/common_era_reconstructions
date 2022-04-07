import sys,os,copy

import numpy as np
import pickle

from time import time
from spharm import Spharmt, getspecindx, regrid
from netCDF4 import Dataset
from scipy import stats

sys.path.insert(1,'/home/disk/kalman2/mkb22/LMR_lite/')
import LMR_lite_utils as LMRlite
import LMR_utils as lmr
import LMR_config

sys.path.insert(1,'/home/disk/p/mkb22/Documents/si_utils_kb/')
import Sice_utils as siutils 

sys.path.insert(1,'/home/disk/p/mkb22/Documents/si_analysis_kb/common_era_experiments/')
import commonera_utils as ceutils 

#--------------------------------------------------------------------
# USER PARAMETERS: 
#--------------------------------------------------------------------
# inflate the sea ice variable here (can inflate whole state here too)
inflate = 1.0
inf_name = '1'

#prior_name = 'mpi'
prior_name = 'ccsm4'

pp_err = 0.4
#pp_err = False
serr = '0_4'
#serr = 'pages2kv2'

cfile = '/home/disk/kalman2/mkb22/LMR_lite/configs/config_ccsm4_pseudo_brennan2020.yml'

proxies = 'pseudo'

savedir = ('/home/disk/p/mkb22/nobackup/LMR_output/common_era_experiments/experiments/pseudo/')

print('loading configuration...')
cfg = LMRlite.load_config_simple(cfile)

recon_start = str(cfg.core.recon_period[0])
recon_end = str(cfg.core.recon_period[1])
loc_list = [cfg.core.loc_rad]
    
iter_range = cfg.wrapper.iter_range
MCiters = range(iter_range[0],iter_range[1]+1)

proxy_ind = np.zeros((iter_range[1]+1,405))
prior_ind = np.zeros((iter_range[1]+1,200))
#--------------------------------------------------------------------
# END USER PARAMETERS: 
#--------------------------------------------------------------------

# Start looping over Monte Carlo iterations: 
for iter_num in MCiters:
    savename = ('sic_'+prior_name+'_anrecon_'+str(recon_start)+'_'+
                     str(recon_end)+'_'+proxies+'_inf'+inf_name+'_loc'+str(loc_list[0])+
                    '_R'+serr+'_iter'+str(iter_num)+'.pkl')

    print('Starting iteration '+str(iter_num)+': inflation = '+str(inflate)+', '+inf_name)

    cfg_dict = lmr.param_cfg_update('core.curr_iter',iter_num)

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
    # LOAD PRIOR ENSEMBLE: 
    print('loading prior')
    X, Xb_one = LMRlite.load_prior(cfg)
    Xbp = Xb_one - Xb_one.mean(axis=1,keepdims=True)

    # Save lat/lon of prior for later reloading and regridding of truth data
    prior_lat_full = X.prior_dict['tas_sfc_Amon']['lat']
    prior_lon_full = X.prior_dict['tas_sfc_Amon']['lon']

    prior_sic_lat_full = X.prior_dict['sic_sfc_OImon']['lat']
    prior_sic_lon_full = X.prior_dict['sic_sfc_OImon']['lon']

    # check if config is set to regrid the prior
    if cfg.prior.regrid_method:
        print('regridding prior...')
        # this function over-writes X, even if return is given a different name
        [X_regrid,Xb_one_new] = LMRlite.prior_regrid(cfg,X,Xb_one,verbose=True)
    else:
        X_regrid.trunc_state_info = X.full_state_info

    # --------------------------------------------------
    # make a grid object for the prior
    grid = LMRlite.Grid(X_regrid)

    # locate position of variables in Xb_one_new: 
    tas_pos = X_regrid.trunc_state_info['tas_sfc_Amon']['pos']
    sic_pos = X_regrid.trunc_state_info['sic_sfc_OImon']['pos']

    # --------------------------------------------------
    # RELOAD FULL PRIORS TO BE CONSIDERED TRUTH: 

    # Load full prior for tas and regrid
    prior_tas_orig = X.prior_dict['tas_sfc_Amon']['value']
    nyears = prior_tas_orig.shape[0]
    
    prior_regrid_tas_prep = np.transpose(np.reshape(prior_tas_orig,
                                                    (prior_tas_orig.shape[0],       
                                                     prior_tas_orig.shape[1]*prior_tas_orig.shape[2])))

    # Regrid full tas prior for proxy selection and verification: 
    [prior_tas_regrid,lat_tas_new,lon_tas_new] = lmr.regrid_esmpy(cfg.prior.esmpy_grid_def['nlat'],
                                                            cfg.prior.esmpy_grid_def['nlon'],
                                                            nyears,
                                                            prior_regrid_tas_prep,
                                                            prior_lat_full,
                                                            prior_lon_full,
                                                            prior_lat_full.shape[0],
                                                            prior_lat_full.shape[1],
                                                            method=cfg.prior.esmpy_interp_method)

    prior_tas_regrid = np.reshape(np.transpose(prior_tas_regrid),(nyears,
                                                                  cfg.prior.esmpy_grid_def['nlat'],
                                                                  cfg.prior.esmpy_grid_def['nlon'],))

    # --------------------------------------------------
    # INFLATION:
    # inflate the entire state vector
    if 2 == 1:
        print('inflating full state vector by '+str(inflate))
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

    # --------------------------------------------------
    # load proxies
    prox_manager = LMRlite.load_proxies(cfg)

    # NH surface area in M km^2 from concentration in percentage
    nharea = 2*np.pi*(6380**2)/1e8

    # Loop over all years available in this reference dataset
    recon_start = cfg.core.recon_period[0]
    recon_end = cfg.core.recon_period[1]
    recon_years = range(recon_start,recon_end)
    #recon_years = range(1979,2000)
    nyears = len(recon_years)

    # Find recon time indices in prior 
    prior_time = np.arange(850,1851,1)
    recon_ind_start = np.min(np.where(prior_time>=recon_start))
    recon_ind_end = np.max(np.where(prior_time<=recon_end))

    # --------------------------------------------------
    # PREPARE TRUTH DATA PSEUDO OBSERVATIONS WILL BE DRAWN FROM:

    # Prior anomalies to draw pseudo proxies from: 
    if cfg.prior.state_variables['tas_sfc_Amon'] == 'full':
        print('reporting true TAS values as full fields')
        prior_tas_regrid_true = prior_tas_regrid[recon_ind_start:recon_ind_end,:,:]
    else: 
        print('reporting true TAS values as anomalies')
        pmean = np.nanmean(prior_tas_regrid[recon_ind_start:recon_ind_end,:,:],axis=0)
        prior_tas_regrid_true = (prior_tas_regrid - pmean)[recon_ind_start:recon_ind_end,:,:]

#    sic_prior_regrid_true = sic_prior_regrid[recon_ind_start:recon_ind_end,:,:]

    # Break up prior by two variables 
    Xb_one_tas = Xb_one_new[tas_pos[0]:tas_pos[1]+1,:]
    Xb_one_sic = Xb_one_new[tas_pos[1]+1:Xb_one.shape[0],:]

    # this is used for drawing prior ensemble estimates of pseudoproxy (Ye)
    Xb_sampler = np.reshape(Xb_one_tas,[grid.nlat,grid.nlon,grid.nens])
    print('pseduoproxy sampler shape:',Xb_sampler.shape)

    # make a grid object for the reference data
    grid_pseudo = LMRlite.Grid()
    pseudolon,pseudolat = np.meshgrid(grid.lon[0,:],grid.lat[:,0])
    grid_pseudo.lat = pseudolat
    grid_pseudo.lon = pseudolon
    grid_pseudo.nlat = grid_pseudo.lat.shape[0]
    grid_pseudo.nlon = grid_pseudo.lon.shape[1]

    # --------------------------------------------------
    # INITIALIZE THE RECONSTRUCTION: 
    # --------------------------------------------------
    sic_save = []
    obs_size = np.zeros((nyears))
    obs_full = {}
    sic_save_lalo = np.zeros((nyears,grid.nlat,grid.nlon))
    tas_save_lalo = np.zeros((nyears,grid.nlat,grid.nlon))
    sic_lalo_full = np.zeros((grid.nlat,grid.nlon,grid.nens))
    var_save = []
    sic_full_Nens = []
    sie_full_Nens = []
    sic_full_Sens = []
    sie_full_Sens = []
    sic_ndim = sic_pos[1]-sic_pos[0]+1
    prox_lat = {}
    prox_lon = {}
    prox_err = {}

    begin_time = time()
    yk = -1
    for yk, target_year in enumerate(recon_years):
        print('Reconstructing year '+str(target_year))

        # Get list of proxy locations for this year: 
        for proxy_idx, Y in enumerate(prox_manager.sites_assim_proxy_objs()):
            if target_year in Y.time: 
                if proxy_idx is 0:
                    prox_lat[target_year] = []
                    prox_lon[target_year] = []
                    prox_err[target_year] = []
                    prox_lat[target_year] = np.append(prox_lat[target_year],Y.lat)
                    prox_lon[target_year] = np.append(prox_lon[target_year],Y.lon)
                    prox_err[target_year] = np.append(prox_err[target_year],Y.psm_obj.R)
                else: 
                    prox_lat[target_year] = np.append(prox_lat[target_year],Y.lat)
                    prox_lon[target_year] = np.append(prox_lon[target_year],Y.lon)
                    prox_err[target_year] = np.append(prox_err[target_year],Y.psm_obj.R)

        nobs = prox_lat[target_year].shape[0]
        obs_size[yk] = nobs

        vY = np.zeros((nobs))
        vYe = np.zeros([nobs,grid.nens])
        vYe_coords = []
        vR = []
        k = -1
        obs_loc = []

        print('Number of obs for this year is = '+str(nobs))

        # Draw pseudo observations from proxy locations for this recon year:
        for iob in range(nobs):
    #        print('assimilating ob '+str(i))
            if pp_err is False: 
                p_err = prox_err[target_year][iob]
            else: 
                p_err = pp_err
                
            [pseudoproxy, 
             itlat, itlon] = ceutils.draw_pseudoproxy(grid_pseudo, 
                                                      prox_lat[target_year][iob],
                                                      prox_lon[target_year][iob],
                                                      prior_tas_regrid_true[yk,:,:],p_err)

            vY[iob] = pseudoproxy
            vYe[iob,:] = Xb_sampler[itlat,itlon,:]
            vR.append(p_err)
            obs_loc = np.append(obs_loc,[grid_pseudo.lat[itlat,0],
                                         grid_pseudo.lon[0,itlon]])

            # make vector of Ye coordinates for localization
            if iob is 0: 
                vYe_coords = np.array([grid_pseudo.lat[itlat,0],
                                       grid_pseudo.lon[0,itlon]])[np.newaxis,:]
            else: 
                new = np.array([grid_pseudo.lat[itlat,0],grid_pseudo.lon[0,itlon]])[np.newaxis,:]
                vYe_coords = np.append(vYe_coords,new, axis=0)

        obs_full[target_year] = np.reshape(obs_loc,(nobs,2))

        # Do data assimilation
        #xam,Xap,_ = LMRlite.Kalman_optimal(obs_QC,R_QC,Ye_QC,Xb_one_inflate)
        xam,Xap = LMRlite.Kalman_ESRF(cfg,vY,vR,vYe,
                                      Xb_one_inflate,X=X_regrid,
                                      vYe_coords=vYe_coords,verbose=False)

        # Save full fields and total area for later. 
        tas_lalo = np.reshape(xam[tas_pos[0]:tas_pos[1]+1,np.newaxis]+Xap[tas_pos[0]:tas_pos[1]+1],
                              [grid.nlat,grid.nlon,grid.nens])
        tas_save_lalo[yk,:,:] = np.mean(tas_lalo,axis=2)

        # this saves sea-ice area for the entire ensemble
        sic_Nens = []
        sie_Nens = []
        sic_Sens = []
        sie_Sens = []

        #   for k in range(grid.nens):
        sic_lalo = np.reshape(xam[sic_pos[0]:sic_pos[1]+1,np.newaxis]+Xap[sic_pos[0]:sic_pos[1]+1,:],
                              [grid.nlat,grid.nlon,grid.nens])
        if 'full' in cfg.prior.state_variables['sic_sfc_OImon']:
            sic_lalo = np.where(sic_lalo<0.0,0.0,sic_lalo)
            sic_lalo = np.where(sic_lalo>100.0,100.0,sic_lalo)

            # Calculate extent: 
            sie_lalo = siutils.calc_sea_ice_extent(sic_lalo,15.0)
        else: 
            sic_lalo = sic_lalo
            sie_lalo = np.zeros(sic_lalo.shape)

        for k in range(grid.nens):
            _,nhmic,shmic= lmr.global_hemispheric_means(sic_lalo[:,:,k],grid.lat[:, 0])
            _,sie_nhmic,sie_shmic = lmr.global_hemispheric_means(sie_lalo[:,:,k],grid.lat[:, 0])
            sic_Nens.append(nhmic)
            sie_Nens.append(sie_nhmic)
            sic_Sens.append(shmic)
            sie_Sens.append(sie_shmic)

        sic_save_lalo[yk,:,:] = np.nanmean(sic_lalo,axis=2)
        var_save.append(np.var(sic_Nens,ddof=1))
        sic_full_Nens.append(sic_Nens)
        sie_full_Nens.append(sie_Nens)
        sic_full_Sens.append(sic_Sens)
        sie_full_Sens.append(sie_Sens)

        print('done reconstructing: ',target_year)

    elapsed_time = time() - begin_time
    print('-----------------------------------------------------')
    print('completed in ' + str(elapsed_time) + ' seconds')
    print('-----------------------------------------------------')

    sic_recon = {}
    sic_recon['sic_lalo'] = sic_save_lalo
    sic_recon['tas_lalo'] = tas_save_lalo
    sic_recon['sic_ens_var'] = np.mean(var_save)
    sic_recon['nobs'] = obs_size
    sic_recon['obs_loc'] = obs_full
    sic_recon['sia_Nens'] = np.squeeze(np.array(sic_full_Nens))*nharea
    sic_recon['sie_Nens'] = np.squeeze(np.array(sie_full_Nens))*nharea
    sic_recon['sia_Sens'] = np.squeeze(np.array(sic_full_Sens))*nharea
    sic_recon['sie_Sens'] = np.squeeze(np.array(sie_full_Sens))*nharea
    sic_recon['recon_years'] = recon_years
#    sic_recon['Ye_assim'] = Ye_assim
#    sic_recon['Ye_assim_coords'] = Ye_assim_coords
    sic_recon['Xb_inflate'] = Xb_one_inflate
    sic_recon['prox_lat'] = prox_lat
    sic_recon['prox_lon'] = prox_lon
    sic_recon['tas_truth'] = prior_tas_regrid_true

    #-----------------------------------------------------
    # SAVE OUTPUT: 
    print('Saving experiment to: ',savedir+savename)
    pickle.dump(sic_recon,open(savedir+savename, "wb"))
    #-----------------------------------------------------





