import sys,os,copy

import LMR_lite_utils as LMRlite
import sys
import LMR_config_greg
import numpy as np
import pickle

from time import time
from spharm import Spharmt, getspecindx, regrid
from netCDF4 import Dataset
from scipy import stats

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.util import add_cyclic_point

import reanalysis_recons_utils as rrutils

sys.path.insert(1,'/home/disk/kalman2/mkb22/pyLMR/')
import LMR_utils 

sys.path.insert(1,'/home/disk/p/mkb22/Documents/si_utils_kb/')
import Sice_utils as siutils 

#USER PARAMETERS: -----------------------------------
# reconstruct sea ice using these instrumental temperature datasets
dset_chosen = ['GIS', 'BE', 'CRU']
#dset_chosen = ['CRU']

annual_mean = True

# define an even (lat,lon) grid for ob locations:
#dlat = 20.
dlat = 10.
#dlat = 5.
dlon = dlat

# set the ob error variance (uniform for now)
#r = 0.001
#r = 0.1
r = 0.4
#r= 10.

r_nm = '4'

# inflate the sea ice variable here (can inflate whole state here too)
#inflate = 1.#78
inflate_list = [2,2.5,3]

savedir = '/home/disk/p/mkb22/nobackup/LMR_output/reanalysis_reconstruction_data/annual/'
# savename = ['sic_mpi_mo9_recon_1850_2018_gis_cru_be_R0_25_10deg_inf2_5_e.pkl',
#             'sic_mpi_mo9_recon_1850_2018_gis_cru_be_R0_25_10deg_inf3_e.pkl',
#             'sic_mpi_mo9_recon_1850_2018_gis_cru_be_R0_25_10deg_inf3_5_e.pkl',
#             'sic_mpi_mo9_recon_1850_2018_gis_cru_be_R0_25_10deg_inf4_e.pkl',
#             'sic_mpi_mo9_recon_1850_2018_gis_cru_be_R0_25_10deg_inf4_5_e.pkl',
#             'sic_mpi_mo9_recon_1850_2018_gis_cru_be_R0_25_10deg_inf5_e.pkl',
#             'sic_mpi_mo9_recon_1850_2018_gis_cru_be_R0_25_10deg_inf5_5_e.pkl']

savename1 = ['sic_tas_ccsm4_annual_recon_loc10000_1850_2018_gis_cru_be_R0_'+r_nm+'_10deg_inf2_',
             'sic_tas_ccsm4_annual_recon_loc10000_1850_2018_gis_cru_be_R0_'+r_nm+'_10deg_inf2_5_',
             'sic_tas_ccsm4_annual_recon_loc10000_1850_2018_gis_cru_be_R0_'+r_nm+'_10deg_inf3_']
#              'sic_mpi_annual_recon_noloc_1850_2018_gis_cru_be_R0_'+r_nm+'_10deg_inf2_',
#              'sic_mpi_annual_recon_noloc_1850_2018_gis_cru_be_R0_'+r_nm+'_10deg_inf2_5_',
#              'sic_mpi_annual_recon_noloc_1850_2018_gis_cru_be_R0_'+r_nm+'_10deg_inf3_']
#             'sic_mpi_mo'+str(MONTH)+'_recon2_1850_2018_gis_cru_be_R0_'+r_nm+'_10deg_inf4_',
#             'sic_mpi_mo'+str(MONTH)+'_recon2_1850_2018_gis_cru_be_R0_'+r_nm+'_10deg_inf5_']

alpha = ['b','c','d','e']

for letter in alpha:
    #----------------------------------------------------
    print('loading configuration...')

    #cfile = './config/config.yml.instrumental'
    #cfile = './config/config.yml.instrumental_sea_ice'
    cfile = './configs/config.yml.test_annual'
    yaml_file = os.path.join(LMR_config_greg.SRC_DIR,cfile)
    cfg = LMRlite.load_config(yaml_file)

    # automatically detect if the config is annual-mean or other
    if cfg.core.recon_months == list(range(1,13,1)):
        annual_mean = True

    if annual_mean:
        print('reconstructing annual-mean values')
        time_label='annual'
    else:
        print('reconstructing values averaged over months',cfg.core.recon_months)
        time_label = 'months'
        for k in cfg.core.recon_months:
            time_label = time_label+'-'+str(k)
        print('time_label=',time_label)

    print('loading prior')
    X, Xb_one = LMRlite.load_prior(cfg)
    Xbp = Xb_one - Xb_one.mean(axis=1,keepdims=True)

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

    # make a grid object for the prior
    grid = LMRlite.Grid(X_regrid)

    # locate 2m air temperature in Xb_one and make a new array
    tas_pos = X_regrid.trunc_state_info['tas_sfc_Amon']['pos']
    tas = Xb_one_new[tas_pos[0]:tas_pos[1]+1,:]

    # isolate sea-ice concentration
    # this is a hack because the variable names are different---FIX!
    # if prior_id_string == 'ccsm4_last_millenium':
    #     print('assigning '+prior_id_string+ ' sea ice ...')
    #     sic_pos = X_regrid.trunc_state_info['sic_noMV_OImon']['pos']
    # else:
    #     sic_pos = X_regrid.trunc_state_info['sic_sfc_OImon']['pos']
    #     print('assigning '+prior_id_string+ ' sea ice ...')

    # fix using file system softlink for ccsm4 filename...
    sic_pos = X_regrid.trunc_state_info['sic_sfc_OImon']['pos']
    print('assigning '+prior_id_string+ ' sea ice ...')

    sic = Xb_one_new[sic_pos[0]:sic_pos[1]+1,:]

    # load analyses and make "observations" 
    if annual_mean:
        [analysis_data,
        analysis_time,
        analysis_lat,
        analysis_lon] = LMRlite.load_analyses(cfg,full_field=True,outfreq='annual')
    else:
        [analysis_data_month,
        analysis_time_month,
        analysis_lat,
        analysis_lon] = LMRlite.load_analyses(cfg,full_field=True,outfreq='monthly')

    if annual_mean:
        # TEMPORARY FIX FOR ANNUAL RECONS: 
        gis_time_mo = np.reshape(analysis_time['GIS'][0:1644],(137,12))
        gis_time_yr = gis_time_mo[:,1]
        gis_data_mo = np.reshape(analysis_data['GIS'][0:1644],(137,12,90,180))
        gis_annual = np.nanmean(gis_data_mo,axis=1)

        # Correct GIS data to be annual averages: 
        analysis_time['GIS'] = gis_time_yr
        analysis_data['GIS']= gis_annual

    if not annual_mean:
        print('averaging over months chosen in config for ',dset_chosen,'...')

        analysis_time = {}
        analysis_data = {}
        for ref_dset in dset_chosen:
            print('working on ',ref_dset)

            dset_years = []
            nmo = len(analysis_time_month[ref_dset])
            nyr = int(nmo/12)
            print("nmo,nyr",nmo,nyr)
            dset_data = np.zeros([nyr,analysis_lat[ref_dset].shape[0],analysis_lon[ref_dset].shape[0]])
            kk = -1
            for k in range(0,nyr*12,12):
                kk += 1
                #print(analysis_time_month[ref_dset][k].year,analysis_time_month[ref_dset][k].month)
                # use this for testing that correct months are chosen
                temp = [analysis_time_month[ref_dset][k+i-1].month for i in cfg.core.recon_months]
                # this is the average over the chosen months for the year represented by k
                mmean = np.mean([analysis_data_month[ref_dset][k+i-1,:,:] for i in cfg.core.recon_months],0)
                dset_years.append(analysis_time_month[ref_dset][k].year)
                dset_data[kk,:,:] = mmean

            analysis_time[ref_dset] = np.array(dset_years)
            analysis_data[ref_dset] = dset_data

    #--------------------------------------------------------
    # start reconstruction experiments here for ALL reference datasets
    #--------------------------------------------------------
    # set parameters here

    # NH surface area in M km^2 from concentration in percentage
    nharea = 2*np.pi*(6380**2)/1e8

    for i,inflate in enumerate(inflate_list):
        print('working on inflation value: ',inflate)
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

        # Takes reference data and selects out observations based on dlat (set at beginning)
        # Currently all longitude values are used and every dlat latitude values are used 

        # select the reference dataset to use (['GIS', 'CRU', 'BE', 'MLOST'])
        sic_dict = {}
        sic_dict_lalo = {}
        sic_ens_full = {}
        tas_dict_lalo = {}
        var_dict = {}
        obs_full = {}
        obs_size = {}
        Ye_full = {}
        conf_dict_97_5 = {}
        conf_dict_2_5 = {}
        recon_years_all={}
        sic_ens_dict = {}
        obs_QC_full = {}
        obs_QC_tot = []

        ndset = len(dset_chosen)

        for ref_dset in dset_chosen:

            print('working on: '+ref_dset)

            ref_dat = analysis_data[ref_dset]
            ref_lat = analysis_lat[ref_dset]
            ref_lon = analysis_lon[ref_dset]

            # GMT for all years in the reference data
            ref_gmt, nhmt, shmt = LMR_utils.global_hemispheric_means(ref_dat,ref_lat)

            # make a grid object for the reference data
            grid_ref = LMRlite.Grid()
            reflon,reflat = np.meshgrid(ref_lon,ref_lat)
            grid_ref.lat = reflat
            grid_ref.lon = reflon
            grid_ref.nlat = ref_dat.shape[1]
            grid_ref.nlon = ref_dat.shape[2]

            # make obs from a reference dataset

            # avoid the poles, and insure NH points are symmetric with SH
            ob_lat = np.arange(dlat-90.,90.1-dlat,dlat)
            #ob_lat = np.arange(60.,90.1-dlat,dlat)
            ob_lon = np.arange(0.,360.,dlon)

            # this function makes the obs specified by the ob_lat,ob_lon coords for all times
            obs,_,_ = LMRlite.make_obs(ob_lat,ob_lon,ref_lat,ref_lon,ref_dat,True)
            obs_full[ref_dset] = obs
            nobs = len(obs)

            # select the error associated with the observations
        #    R,_,_ = LMRlite.make_obs(ob_lat,ob_lon,ref_lat,ref_lon,R_cru_annual,True)
        #     R_nan = R
        #     R_nan = np.where(R_nan<0.0,np.nan,R_nan)

            # ob error variance vector
            R = r*np.ones(nobs)
            #R = np.array(R_all[ref_dset])
        #    R = Rvar

            # make ob estimates (Ye) from the prior (time,lat,lon)
            tas = Xb_one_inflate[tas_pos[0]:tas_pos[1]+1,:]
            tmp = np.reshape(tas,[grid.nlat,grid.nlon,grid.nens])
            Xb_ye = np.rollaxis(tmp,2,0)
            Ye,Ye_ind_lat,Ye_ind_lon = LMRlite.make_obs(ob_lat,ob_lon,grid.lat[:,0],grid.lon[0,:],Xb_ye,True)
            Ye_full[ref_dset] = Ye
            
           # make vector of Ye coordinates for localization
            Ye_ind_lon_2d = Ye_ind_lon[:,np.newaxis]
            Ye_ind_lat_2d = Ye_ind_lat[:,np.newaxis]
            vYe_coords = np.append(Ye_ind_lat_2d,Ye_ind_lon_2d, axis=1)

            #--------------------------------------------------------------------------------
            # Loop over all years available in this reference dataset
            recon_years = range(min(analysis_time[ref_dset]),max(analysis_time[ref_dset]))
#             recon_years = range(2000,max(analysis_time[ref_dset]))
            recon_years_all[ref_dset] = recon_years
        #     recon_years = overlap_pd
        #     recon_years_all[ref_dset] = overlap_pd

            # Option for a custom range of years
            #recon_years = range(1970,2018)

            nyears = len(recon_years)

            # These are the time indices for the reference dataset; useful later
            iyr_ref = np.zeros(nyears,dtype=np.int16)

            sic_save = []
            sic_save_lalo = np.zeros((nyears,grid.nlat,grid.nlon))
            tas_save_lalo = np.zeros((nyears,grid.nlat,grid.nlon))
            var_save = []
            conf97_5_save = []
            conf2_5_save = []
            sic_full_ens = []
            sic_ndim = sic_pos[1]-sic_pos[0]+1

            # Option to save the full ensemble
            #Xap_save = np.zeros([nyears,sic_ndim,cfg.core.nens])
            #Xap_var_save = np.zeros([nyears,sic_ndim])

            obs_shape = np.zeros((nyears))

            begin_time = time()
            yk = -1
            for yk, target_year in enumerate(recon_years):
                iyr = np.argwhere(analysis_time[ref_dset]==target_year)[0][0]
                iyr_ref[yk] = int(iyr)

                # QC nan obs (argwhere gives indices)
                qcpass1 = np.argwhere(np.isfinite(obs[:,iyr]))
                obs_QC = np.squeeze(obs[qcpass1,iyr])
                #R_QC = np.squeeze(R[qcpass1,iyr])
                #R_QC = np.squeeze(R[qcpass1,yk])
                R_QC = np.squeeze(R[qcpass1])
                Ye_QC = np.squeeze(Ye[qcpass1,:])
                vYe_coords_QC = np.squeeze(vYe_coords[qcpass1,:])

                # QC nan in R (argwhere gives indices)
    #             qcpass_R = np.argwhere(np.isfinite(R_QC1))
    #             obs_QC = np.squeeze(obs_QC1[qcpass_R])
    #             R_QC = np.squeeze(R_QC1[qcpass_R])+0.00000000001
    #             Ye_QC = np.squeeze(Ye_QC1[qcpass_R])

                obs_shape[yk] = obs_QC.size
                obs_QC_tot.append(obs_QC)

                # Do data assimilation
#                xam,Xap,_ = LMRlite.Kalman_optimal(obs_QC,R_QC,Ye_QC,Xb_one_inflate)
                xam,Xap = LMRlite.Kalman_ESRF(cfg,obs_QC,R_QC,Ye_QC,
                                              Xb_one_inflate,X=X_regrid,
                                              vYe_coords=vYe_coords_QC,verbose=False)

                tas_lalo = np.reshape(xam[tas_pos[0]:tas_pos[1]+1],[grid.nlat,grid.nlon])
                sic_lalo = np.reshape(xam[sic_pos[0]:sic_pos[1]+1],[grid.nlat,grid.nlon])
        #         sic_lalo_pc = sic_lalo[sic_lalo<0.0]=0.0
        #         sic_lalo_pc = sic_lalo[sic_lalo>100.0]=100.0
                _,nhmic,_ = LMR_utils.global_hemispheric_means(sic_lalo,grid.lat[:, 0])
                sic_save.append(nhmic)
                sic_save_lalo[yk,:,:] = sic_lalo
                tas_save_lalo[yk,:,:] = tas_lalo

                # this saves sea-ice area for the entire ensemble
                sic_ens = []
                for k in range(grid.nens):
#                     tas_lalo = np.reshape(xam[tas_pos[0]:tas_pos[1]+1]+Xap[tas_pos[0]:tas_pos[1]+1,k],
#                                           [grid.nlat,grid.nlon])
                    sic_lalo = np.reshape(xam[sic_pos[0]:sic_pos[1]+1]+Xap[sic_pos[0]:sic_pos[1]+1,k],
                                          [grid.nlat,grid.nlon])
        #             sic_lalo_pc = sic_lalo[sic_lalo<0.0] = 0.0
        #             sic_lalo_pc = sic_lalo[sic_lalo>100.0] = 100.0
                    _,nhmic,_ = LMR_utils.global_hemispheric_means(sic_lalo,grid.lat[:, 0])
                    sic_ens.append(nhmic)
                var_save.append(np.var(sic_ens,ddof=1))
                conf97_5_save.append(np.percentile(sic_ens,97.5))
                conf2_5_save.append(np.percentile(sic_ens,2.5))
                sic_full_ens.append(sic_lalo)

                # this saves the gridded concentration field for the entire ensemble
                #Xap_save[0,:,:] = Xap[sic_pos[0]:sic_pos[1]+1]
                #Xap_var = np.var(Xap[sic_pos[0]:sic_pos[1]+1,:],axis=1,ddof=1)
                #Xap_var_save[0,:] = Xap_var
                print(target_year)

            elapsed_time = time() - begin_time
            print('-----------------------------------------------------')
            print('completed in ' + str(elapsed_time) + ' seconds')
            print('-----------------------------------------------------')
            sic_dict[ref_dset] = sic_save
            sic_dict_lalo[ref_dset] = sic_save_lalo
            sic_ens_full[ref_dset] = sic_full_ens
            tas_dict_lalo[ref_dset] = tas_save_lalo
            var_dict[ref_dset] = np.mean(var_save)
            obs_size[ref_dset] = obs_shape
            sic_ens_dict[ref_dset] = np.squeeze(sic_ens)
            conf_dict_97_5[ref_dset] = conf97_5_save  
            conf_dict_2_5[ref_dset] = conf2_5_save
            obs_QC_full[ref_dset] = obs_QC_tot

        sic_recon = {}
        sic_recon['sic_dict'] = sic_dict
        sic_recon['sic_dict_lalo'] = sic_dict_lalo
        sic_recon['tas_dict_lalo'] = tas_dict_lalo
        sic_recon['sic_ens_full'] = sic_ens_full
        #sic_recon['sic_full_ens'] = sic_full_ens
        sic_recon['var_dict'] = var_dict
        sic_recon['sic_conf_97_5'] = conf_dict_97_5
        sic_recon['sic_conf_2_5'] = conf_dict_2_5
        sic_recon['obs_full'] = obs_full
        sic_recon['obs_size'] = obs_size
        sic_recon['recon_years_all'] = recon_years_all
        sic_recon['R_all'] = R_QC
        sic_recon['Ye_full'] = Ye_full 

        pickle.dump(sic_recon,open(savedir+savename1[i]+letter+'.pkl', "wb"))
        print('Saved to: '+savedir+savename1[i]+letter+'.pkl')
