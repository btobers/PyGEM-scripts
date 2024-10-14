"""Run model calibration."""

# Built-in libraries
import argparse
import inspect
import multiprocessing
import os
import sys
import time
import math
import warnings
# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import pygem
import pygem_input as pygem_prms
from pygem import mcmc
from pygem import class_climate
from pygem.massbalance import PyGEMMassBalance
#from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory, single_flowline_glacier_directory_with_calving
import pygem.pygem_modelsetup as modelsetup
from pygem.shop import debris, mbdata, icethickness, surfelev

from oggm import cfg
from oggm import graphics
from oggm import tasks
from oggm import utils
from oggm import workflow
from oggm.core.flowline import FluxBasedModel
from oggm.core.massbalance import apparent_mb_from_any_mb
#from oggm.core import climate
#from oggm.core.flowline import FluxBasedModel
#from oggm.core.inversion import calving_flux_from_depth

import torch
import gpytorch
import sklearn.model_selection

#%% FUNCTIONS
def getparser():
    """
    Use argparse to add arguments from the command line

    Parameters
    ----------
    ref_gcm_name (optional) : str
        reference gcm name
    num_simultaneous_processes (optional) : int
        number of cores to use in parallels
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers which is used to run batches on the supercomputer
    rgi_glac_number : str
        rgi glacier number to run for supercomputer
    progress_bar : bool
        Switch for turning the progress bar on or off (default = False)
    debug : bool
        Switch for turning debug printing on or off (default = False)

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='reference gcm name')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc'),
    parser.add_argument('-rgi_glac_number', action='store', type=str, default=pygem_prms.glac_no, nargs='+',
                        help='rgi glacier number for supercomputer')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use (default is 1, ie. no parallelization)')
    # flags
    parser.add_argument('-p', '--progress_bar', action='store_true',
                        help='Flag to show progress bar')
    parser.add_argument('-v', '--debug', action='store_true',
                        help='Flag for debugging')
    return parser


def mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=None, t1=None, t2=None,
                 option_areaconstant=1, return_tbias_mustmelt=False, return_tbias_mustmelt_wmb=False):
    """
    Run the mass balance and calculate the mass balance [mwea]

    Parameters
    ----------
    option_areaconstant : Boolean

    Returns
    -------
    mb_mwea : float
        mass balance [m w.e. a-1]
    """
    # RUN MASS BALANCE MODEL
    mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table, fls=fls, option_areaconstant=True,
                             debug=pygem_prms.debug_mb, debug_refreeze=pygem_prms.debug_refreeze)
    years = np.arange(0, int(gdir.dates_table.shape[0]/12))
    for year in years:
        mbmod.get_annual_mb(fls[0].surface_h, fls=fls, fl_id=0, year=year, debug=False)
    
    # Option for must melt condition
    if return_tbias_mustmelt:
        # Number of years and bins with negative climatic mass balance
        nbinyears_negmbclim =  len(np.where(mbmod.glac_bin_massbalclim_annual < 0)[0])
        return nbinyears_negmbclim
    elif return_tbias_mustmelt_wmb:
        nbinyears_negmbclim =  len(np.where(mbmod.glac_bin_massbalclim_annual < 0)[0])
        t1_idx = gdir.mbdata['t1_idx']
        t2_idx = gdir.mbdata['t2_idx']
        nyears = gdir.mbdata['nyears']
        mb_mwea = mbmod.glac_wide_massbaltotal[t1_idx:t2_idx+1].sum() / mbmod.glac_wide_area_annual[0] / nyears
        return nbinyears_negmbclim, mb_mwea
    # Otherwise return specific mass balance
    else:        
        # Specific mass balance [mwea]
        t1_idx = gdir.mbdata['t1_idx']
        t2_idx = gdir.mbdata['t2_idx']
        nyears = gdir.mbdata['nyears']
        mb_mwea = mbmod.glac_wide_massbaltotal[t1_idx:t2_idx+1].sum() / mbmod.glac_wide_area_annual[0] / nyears
        return mb_mwea


def get_binned_dh(gdir, modelprms, glacier_rgi_table, fls=None, glen_a_multiplier=None, fs=None, time_inds=None, bin_edges=None, debug=False):
    """
    Run the ice thickness inversion and mass balance model to get binned annual ice thickness evolution
    Convert to monthly thickness by assuming that the flux divergence is constant throughout the year
    """
    nyears = int(gdir.dates_table.shape[0]/12) # number of years from dates table
    # perform OGGM ice thickness inversion
    if not gdir.is_tidewater or not pygem_prms.include_calving:
        # Perform inversion based on PyGEM MB using reference directory
        mbmod_inv = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                        hindcast=pygem_prms.hindcast,
                                        debug=pygem_prms.debug_mb,
                                        debug_refreeze=pygem_prms.debug_refreeze,
                                        fls=fls, option_areaconstant=True,
                                        inversion_filter=pygem_prms.include_debris)
        
        # Arbitrariliy shift the MB profile up (or down) until mass balance is zero (equilibrium for inversion)
        apparent_mb_from_any_mb(gdir, mb_years=np.arange(nyears), mb_model=mbmod_inv)
        tasks.prepare_for_inversion(gdir)
        tasks.mass_conservation_inversion(gdir, glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs)
        tasks.init_present_time_glacier(gdir) # adds bins below
        debris.debris_binned(gdir, fl_str='model_flowlines') # add debris enhancement factors to flowlines
        try:
            nfls = gdir.read_pickle('model_flowlines')
        except FileNotFoundError as e:
            if 'model_flowlines.pkl' in str(e):
                tasks.compute_downstream_line(gdir)
                tasks.compute_downstream_bedshape(gdir)
                tasks.init_present_time_glacier(gdir) # adds bins below
                nfls = gdir.read_pickle('model_flowlines')
            else:
                raise

        # Check that water level is within given bounds
        cls = gdir.read_pickle('inversion_input')[-1]
        th = cls['hgt'][-1]
        vmin, vmax = cfg.PARAMS['free_board_marine_terminating']
        water_level = utils.clip_scalar(0, th - vmax, th - vmin) 
        # mass balance model with evolving area
        mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table,
                                    hindcast=pygem_prms.hindcast,
                                    debug=pygem_prms.debug_mb,
                                    debug_refreeze=pygem_prms.debug_refreeze,
                                    fls=nfls, option_areaconstant=False)
        
        # glacier dynamics model
        ev_model = FluxBasedModel(nfls, y0=0, mb_model=mbmod, 
                                    glen_a=cfg.PARAMS['glen_a']*glen_a_multiplier, fs=fs,
                                    is_tidewater=gdir.is_tidewater,
                                    water_level=water_level)
        
        try:
            # run glacier dynamics model forward
            ev_model.run_until_and_store(nyears)
            mb_mwea = mbmod.glac_wide_massbaltotal[gdir.mbdata['t1_idx']:gdir.mbdata['t2_idx']+1].sum() / mbmod.glac_wide_area_annual[0] / nyears

        except RuntimeError:
            return np.nan, np.nan

        # Update the latest thickness
        if ev_model is not None:
            fl_widths_m = getattr(ev_model.fls[0], 'widths_m', None)
            fl_section = getattr(ev_model.fls[0],'section',None)
        else:
            fl_widths_m = getattr(nfls[0], 'widths_m', None)
            fl_section = getattr(nfls[0],'section',None)
        if fl_section is not None and fl_widths_m is not None:                                
            # thickness
            icethickness_t0 = np.zeros(fl_section.shape)
            icethickness_t0[fl_widths_m > 0] = fl_section[fl_widths_m > 0] / fl_widths_m[fl_widths_m > 0]
            mbmod.glac_bin_icethickness_annual[:,-1] = icethickness_t0

        # get annual climatic mass balance from monthly climatic mass balance - requires reshaping monthly binned values and summing every 12 months
        bin_massbalclim_annual = mbmod.glac_bin_massbalclim.reshape(mbmod.glac_bin_massbalclim.shape[0],mbmod.glac_bin_massbalclim.shape[1]//12,-1).sum(2)

        # bin_thick_annual = bin_thick_annual[:,:-1]
        # get change in thickness from previous year for each elevation bin
        delta_thick_annual = np.diff(mbmod.glac_bin_icethickness_annual, axis=-1)

        # get annual binned flux divergence as annual binned climatic mass balance (-) annual binned ice thickness
        # account for density contrast (convert climatic mass balance in m w.e. to m ice)
        flux_div_annual =   (
                    (bin_massbalclim_annual * 
                    (pygem_prms.density_ice / 
                    pygem_prms.density_water)) - 
                    delta_thick_annual)

        # we'll assume the flux divergence is constant througohut the year (is this a good assumption?)
        # ie. take annual values and divide by 12 - repeat monthly values across 12 months
        flux_div_monthly = np.repeat(flux_div_annual / 12, 12, axis=1)

        # get monthly binned change in thickness assuming constant flux divergence throughout the year
        # account for density contrast (convert monthly climatic mass balance in m w.e. to m ice)
        delta_thick_monthly =   (
                    (mbmod.glac_bin_massbalclim * 
                    (pygem_prms.density_ice / 
                    pygem_prms.density_water)) - 
                    flux_div_monthly)

        # get binned monthly thickness = running thickness change + initial thickness
        running_delta_thick_monthly = np.cumsum(delta_thick_monthly, axis=-1)
        bin_thick =  running_delta_thick_monthly + mbmod.glac_bin_icethickness_annual[:,0][:,np.newaxis]
        # only retain specified time steps
        bin_thick = bin_thick[:,time_inds]

        # aggregate model bin thicknesses as desired
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            y_pred = np.column_stack([stats.binned_statistic(x=nfls[0].surface_h, values=x, statistic=np.nanmean, bins=bin_edges)[0] for x in bin_thick.T])
        binned_dh = np.diff(y_pred,axis=1)

        return mb_mwea, binned_dh


# class for Gaussian Process model for mass balance emulator
class ExactGPModel(gpytorch.models.ExactGP):
    """ Use the simplest form of GP model, exact inference """
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# glacier wide mass balance emulator class object
class massbalEmulator:
    def __init__(self, mod, likelihood, X_mean, X_std, y_mean, y_std):
        self.mod = mod
        self.likelihood = likelihood
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        self.y_std = y_std

    # evaluate the emulator for a given set of model paramaters (note, Xtest should be ordered as so: [tbias, kp, ddfsnow])
    def eval(self, Xtest):
        # normalize each parameter
        Xtest[:] = [(x - mu) / sigma for x, mu, sigma in zip(Xtest, self.X_mean, self.X_std)]
        # convert to torch tensor
        Xtest_normed = torch.tensor(np.array([Xtest])).to(torch.float)
        # pass to mbEmulator.mod() to evaluate normed values
        mb_mwea_norm = self.mod(Xtest_normed).mean.detach().numpy()[0]
        # un-normalize
        mb_mwea = mb_mwea_norm * self.y_std + self.y_mean
        return mb_mwea

    # load emulator
    @classmethod
    def load(cls, em_mod_path=None):
        # ----- LOAD EMULATOR -----
        torch.set_num_threads(1)

        state_dict = torch.load(em_mod_path, weights_only=False)
        emulator_extra_fp = em_mod_path.replace('.pth', '_extra.pkl')
        with open(emulator_extra_fp, 'rb') as f:
            emulator_extra_dict = pickle.load(f)
        # convert lists to torch tensors
        X_train = torch.stack([torch.tensor(lst) for lst in emulator_extra_dict['X_train']], dim=1)
        X_mean = torch.tensor(emulator_extra_dict['X_mean'])
        X_std = torch.tensor(emulator_extra_dict['X_std'])
        y_train = torch.tensor(emulator_extra_dict['y_train'])
        y_mean = torch.tensor(emulator_extra_dict['y_mean'])
        y_std = torch.tensor(emulator_extra_dict['y_std'])
    
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # Create a new GP model
        model = ExactGPModel(X_train, y_train, likelihood)  
        model.load_state_dict(state_dict)
        model.eval()
        
        return cls(model, likelihood, X_mean, X_std, y_mean, y_std)


def create_emulator(glacier_str, sims_df, y_cn, 
                    X_cns=['tbias','kp','ddfsnow'], 
                    em_fp=pygem_prms.output_filepath + 'emulator/', debug=False):
    """
    create emulator for calibrating PyGEM model parameters

    Parameters
    ----------
    glacier_str : str
        glacier RGIId string
    sims_dict : Dictionary
        parameter distributions for each emulator simulation
    y_cn : str
        variable name in sims_dict which to fit parameter sets to fit
    X_cns : list
        PyGEM model parameters to calibrate
    em_fp : str
        filepath to save emulator results

    Returns
    -------
    X_train, X_mean, X_std, y_train, y_mean, y_std, likelihood, model
    """
    # This is required for the supercomputer such that resources aren't stolen from other cpus
    torch.set_num_threads(1)
    
    assert y_cn in sims_df.columns, 'emulator error: y_cn not in sims_df'

    ###################        
    ### get Xy data ###
    ###################
    
    X = sims_df.loc[:,X_cns]
    y = sims_df.loc[:,y_cn]

    if debug:
        print(f'Calibration x-parameters: {", ".join(X_cns)}')
        print(f'Calibration y-parametes: {y_cn}')
        print(f'X:\n{X}')
        print(f'X-shape:\n{X.shape}\n')
        print(f'y:\n{y}')
        print(f'y-shape:\n{y.shape}')
    
    ###################
    # pull values (note order matters here. whenever emulator is evaluated, order should be same as order in X)
    X = X.values
    y = y.values
    # Normalize data
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std = y.std()
    
    y_norm = (y - y_mean) / y_std

    # Split into training and test data and cast to torch tensors
    X_train,X_test,y_train,y_test = [torch.tensor(x).to(torch.float) 
                                        for x in sklearn.model_selection.train_test_split(X_norm,y_norm)]
    # Add a small amount of noise
    y_train += torch.randn(*y_train.shape)*0.01

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X_train, y_train, likelihood)

    # Plot test set predictions prior to training
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():#, gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(X_test))
    idx = np.argsort(y_test.numpy())

    with torch.no_grad():
        lower, upper = y_pred.confidence_region()

        if debug:
            f, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(y_test.numpy()[idx], y_pred.mean.numpy()[idx], 'k*')
            ax.fill_between(y_test.numpy()[idx], lower.numpy()[idx], upper.numpy()[idx], alpha=0.5)
            plt.show()

    # ----- Find optimal model hyperparameters -----
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(1000):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
#        if debug and i%100==0:
#            print(i, loss.item(), model.covar_module.base_kernel.lengthscale[0], 
#                  model.likelihood.noise.item())
        optimizer.step()

    # Plot posterior distributions (with test data on x-axis)
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():#, gpytorch.settings.fast_pred_var():
        y_pred = likelihood(model(X_test))

    idx = np.argsort(y_test.numpy())

    with torch.no_grad():
        lower, upper = y_pred.confidence_region()

        if debug:
            f, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(y_test.numpy()[idx], y_pred.mean.numpy()[idx], 'k*')
            ax.fill_between(y_test.numpy()[idx], lower.numpy()[idx], upper.numpy()[idx], 
                            alpha=0.5)
            plt.show()

    if debug:
        # Compare user-defined parameter sets within the emulator
        tbias_set = (np.arange(-7,4,0.5)).reshape(-1,1)
        kp_set = np.zeros(tbias_set.shape) + 1
        ddf_set = np.zeros(tbias_set.shape) + 0.0041

        modelprms_set = np.hstack((tbias_set, kp_set, ddf_set))
        modelprms_set_norm = (modelprms_set - X_mean) / X_std

        y_set_norm = model(torch.tensor(modelprms_set_norm).to(torch.float)).mean.detach().numpy()
        y_set = y_set_norm * y_std + y_mean

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        kp_1_idx = np.where(sims_df['kp'] == 1)[0]
        ax.plot(sims_df.loc[kp_1_idx,'tbias'], sims_df.loc[kp_1_idx,y_cn])
        ax.plot(tbias_set,y_set,'.')
        ax.set_xlabel('tbias (degC)')
        if y_cn == 'mb_mwea':
            ax.set_ylabel('PyGEM MB (mwea)')
        elif y_cn == 'nbinyrs_negmbclim':
            ax.set_ylabel('nbinyrs_negmbclim (-)')
        plt.show()

        # Compare the modeled and emulated mass balances
        y_em_norm = model(torch.tensor(X_norm).to(torch.float)).mean.detach().numpy()
        y_em = y_em_norm * y_std + y_mean

        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(y,y_em,'.')
        ax.plot([y.min(),y.max()], [y.min(), y.max()])
        if y_cn == 'mb_mwea':
            ax.set_xlabel('emulator MB (mwea)')
            ax.set_ylabel('PyGEM MB (mwea)')
            ax.set_xlim(-1,1)
            ax.set_ylim(-1,1)
        elif y_cn == 'nbinyrs_negmbclim':
            ax.set_xlabel('emulator nbinyrs_negmbclim (-)')
            ax.set_ylabel('PyGEM nbinyrs_negmbclim (-)')
        plt.show()
        
    # ----- EXPORT EMULATOR -----
    # Save emulator (model state, x_train, y_train, etc.)
    em_mod_fn = glacier_str + '-emulator-' + y_cn + '.pth'
    em_mod_fp = em_fp + 'models/' + glacier_str.split('.')[0].zfill(2) + '/'
    if not os.path.exists(em_mod_fp):
        os.makedirs(em_mod_fp, exist_ok=True)
    torch.save(model.state_dict(), em_mod_fp + em_mod_fn)
    # Extra required datasets (convert to lists to avoid any serialization issues with torch tensors)
    em_extra_dict = {'X_train': [X.tolist() for X in X_train.T],
                        'X_mean': [X.tolist() for X in X_mean.T],
                        'X_std': [X.tolist() for X in X_std.T],
                        'y_train': y_train.tolist(),
                        'y_mean': float(y_mean),
                        'y_std': float(y_std)}
    em_extra_fn = em_mod_fn.replace('.pth','_extra.pkl')
    with open(em_mod_fp + em_extra_fn, 'wb') as f:
        pickle.dump(em_extra_dict, f)

    return massbalEmulator(model, likelihood, X_mean, X_std, y_mean, y_std)

    
#%%
def main(list_packed_vars):
    """
    Model simulation

    Parameters
    ----------
    list_packed_vars : list
        list of packed variables that enable the use of parallels

    Returns
    -------
    netcdf files of the simulation output (specific output is dependent on the output option)
    """
    
    # Unpack variables
    glac_no = list_packed_vars[1]
    gcm_name = list_packed_vars[2]
    
    parser = getparser()
    args = parser.parse_args()
    debug = args.debug

    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)

    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.ref_startyear, endyear=pygem_prms.ref_endyear, spinupyears=pygem_prms.ref_spinupyears,
            option_wateryear=pygem_prms.ref_wateryear)

    # ===== LOAD CLIMATE DATA =====
    # Climate class
    assert gcm_name in ['ERA5', 'ERA-Interim'], 'Error: Calibration not set up for ' + gcm_name
    gcm = class_climate.GCM(name=gcm_name)
    # Air temperature [degC]
    gcm_temp, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.temp_fn, gcm.temp_vn, main_glac_rgi, dates_table)
    if pygem_prms.option_ablation == 2 and gcm_name in ['ERA5']:
        gcm_tempstd, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.tempstd_fn, gcm.tempstd_vn,
                                                                        main_glac_rgi, dates_table)
    else:
        gcm_tempstd = np.zeros(gcm_temp.shape)
    # Precipitation [m]
    gcm_prec, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.prec_fn, gcm.prec_vn, main_glac_rgi, dates_table)
    # Elevation [m asl]
    gcm_elev = gcm.importGCMfxnearestneighbor_xarray(gcm.elev_fn, gcm.elev_vn, main_glac_rgi)
    # Lapse rate [degC m-1]
    gcm_lr, gcm_dates = gcm.importGCMvarnearestneighbor_xarray(gcm.lr_fn, gcm.lr_vn, main_glac_rgi, dates_table)
    
    # ===== LOOP THROUGH GLACIERS TO RUN CALIBRATION =====
    for glac in range(main_glac_rgi.shape[0]):
        
        if debug or glac == 0 or glac == main_glac_rgi.shape[0]:
            print(gcm_name,':', main_glac_rgi.loc[main_glac_rgi.index.values[glac],'RGIId'])

        # Select subsets of data
        glacier_rgi_table = main_glac_rgi.loc[main_glac_rgi.index.values[glac], :]
        glacier_str = '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])

        # ===== Load glacier data: area (km2), ice thickness (m), width (km) =====        
        try:
            if not glacier_rgi_table['TermType'] in [1,5] or not pygem_prms.include_calving:
                gdir = single_flowline_glacier_directory(glacier_str, logging_level=pygem_prms.logging_level)
                gdir.is_tidewater = False
            else:
                # set reset=True to overwrite non-calving directory that may already exist
                gdir = single_flowline_glacier_directory_with_calving(glacier_str, logging_level=pygem_prms.logging_level, 
                                                                      reset=False)
                gdir.is_tidewater = True
                
            fls = gdir.read_pickle('inversion_flowlines')
            glacier_area = fls[0].widths_m * fls[0].dx_meter
            
            # Add climate data to glacier directory
            gdir.historical_climate = {'elev': gcm_elev[glac],
                                      'temp': gcm_temp[glac,:],
                                      'tempstd': gcm_tempstd[glac,:],
                                      'prec': gcm_prec[glac,:],
                                      'lr': gcm_lr[glac,:]}
            gdir.dates_table = dates_table

            # ----- Calibration data -----
            try:
            # for batman in [0]:

                mbdata_fn = gdir.get_filepath('mb_obs')

                if not os.path.exists(mbdata_fn):
                    # Compute all the stuff
                        list_tasks = [          
                            # Consensus ice thickness
                            icethickness.consensus_gridded,
                            # Mass balance data
                            mbdata.mb_df_to_gdir
                        ]
                        
                        # Debris tasks
                        if pygem_prms.include_debris:
                            list_tasks.append(debris.debris_to_gdir)
                            list_tasks.append(debris.debris_binned)

                        for task in list_tasks:
                            workflow.execute_entity_task(task, gdir)
            
                with open(mbdata_fn, 'rb') as f:
                    gdir.mbdata = pickle.load(f)
                        
                # Non-tidewater glaciers
                if not gdir.is_tidewater:
                    # Load data
                    mb_obs_mwea = gdir.mbdata['mb_mwea']
                    mb_obs_mwea_err = gdir.mbdata['mb_mwea_err']
                # Tidewater glaciers
                #  use climatic mass balance since calving_k already calibrated separately
                else:
                    mb_obs_mwea = gdir.mbdata['mb_clim_mwea']
                    mb_obs_mwea_err = gdir.mbdata['mb_clim_mwea_err']
                    
                # Add time indices consistent with dates_table for mb calculations
                t1_year = gdir.mbdata['t1_datetime'].year
                t1_month = gdir.mbdata['t1_datetime'].month
                t2_year = gdir.mbdata['t2_datetime'].year
                t2_month = gdir.mbdata['t2_datetime'].month
                t1_idx = dates_table[(t1_year == dates_table['year']) & (t1_month == dates_table['month'])].index.values[0]
                t2_idx = dates_table[(t2_year == dates_table['year']) & (t2_month == dates_table['month'])].index.values[0]
                # Record indices
                gdir.mbdata['t1_idx'] = t1_idx
                gdir.mbdata['t2_idx'] = t2_idx                    
                    
                if debug:
                    print('  mb_data (mwea): ' + str(np.round(mb_obs_mwea,2)) + ' +/- ' + str(np.round(mb_obs_mwea_err,2)))
                
                    
            except:
                gdir.mbdata = None
                
                # LOG FAILURE
                fail_fp = pygem_prms.output_filepath + 'cal_fail/' + glacier_str.split('.')[0].zfill(2) + '/'
                if not os.path.exists(fail_fp):
                    os.makedirs(fail_fp, exist_ok=True)
                txt_fn_fail = glacier_str + "-cal_fail.txt"
                with open(fail_fp + txt_fn_fail, "w") as text_file:
                    text_file.write(glacier_str + ' was missing mass balance data.')
                    
                print('\n' + glacier_str + ' mass balance data missing. Check dataset and column names.\n')

        except:
            fls = None
            
        if debug:
            assert os.path.exists(gdir.get_filepath('mb_obs')), 'Mass balance data missing. Check dataset and column names'


        # oib deltah data
        if pygem_prms.option_calibration == 'MCMC' and pygem_prms.option_calib_binned_dh:
            try:
                # get rgi7id to load oib data
                rgi7id = surfelev.get_rgi7id(glacier_str, debug=debug)
                if rgi7id:
                    oib_dict = surfelev.load_oib(rgi7id)
                    # get oib diffs
                    bin_edges, bin_area, bin_diffs, bin_sigmas, dates = surfelev.get_oib_diffs(oib_dict=oib_dict, aggregate=100)
                    # only retain diffs for survey dates within model timespan
                    _, oib_inds, pygem_inds = np.intersect1d(dates.to_numpy(), gdir.dates_table.date.to_numpy(), return_indices=True)
                    bin_diffs = bin_diffs[:,oib_inds]
                    bin_sigmas = bin_sigmas[:,oib_inds]
                    dates = dates[oib_inds]
                    if debug:
                        print(f'OIB survey dates:\n{", ".join([str(dt.year)+"-"+str(dt.month)+"-"+str(dt.day) for dt in dates])}')
                    # must be at least two surveys
                    if bin_diffs.shape[1] < 2:
                        raise ValueError("Must be at least two individual OIB surveys to difference.")

                    # double difference to remove the COP30 signal from the relative OIB surface elevation changes
                    dbldiffs = np.diff(bin_diffs,axis=1)
                    # take mean sigma_obs from each set of consecutive surveys
                    sigmas = (bin_sigmas[:, :-1] + bin_sigmas[:, 1:]) / 2
                    gdir.deltah = {
                                    'timestamps': dates,
                                    'bin_edges':bin_edges,
                                    'bin_area':bin_area,
                                    'dh':dbldiffs,
                                    'sigma':sigmas
                                }
                    
                    # get glen_a, as dynamics will need to be on to get thickness changes
                    if pygem_prms.use_reg_glena:
                        glena_df = pd.read_csv(pygem_prms.glena_reg_fullfn)                    
                        glena_O1regions = [int(x) for x in glena_df.O1Region.values]
                        assert glacier_rgi_table.O1Region in glena_O1regions, ' O1 region not in glena_df'
                        glena_idx = np.where(glena_O1regions == glacier_rgi_table.O1Region)[0][0]
                        glen_a_multiplier = glena_df.loc[glena_idx,'glens_a_multiplier']
                        fs = glena_df.loc[glena_idx,'fs']
                    else:
                        fs = pygem_prms.fs
                        glen_a_multiplier = pygem_prms.glen_a_multiplier

            except Exception as err:
                fls = None

        # ----- CALIBRATION OPTIONS ------
        if (fls is not None) and (gdir.mbdata is not None) and (glacier_area.sum() > 0):
            
            modelprms = {'kp': pygem_prms.kp,
                        'tbias': pygem_prms.tbias,
                        'ddfsnow': pygem_prms.ddfsnow,
                        'ddfice': pygem_prms.ddfice,
                        'tsnow_threshold': pygem_prms.tsnow_threshold,
                        'precgrad': pygem_prms.precgrad}
            
            #%% ===== EMULATOR TO SETUP MCMC ANALYSIS AND/OR RUN HH2015 WITH EMULATOR =====
            # - precipitation factor, temperature bias, degree-day factor of snow
            if pygem_prms.option_calibration == 'emulator':
                tbias_step = pygem_prms.tbias_step
                tbias_init = pygem_prms.tbias_init
                kp_init = pygem_prms.kp_init
                ddfsnow_init = pygem_prms.ddfsnow_init
                
                # ----- Initialize model parameters -----
                modelprms['tbias'] = tbias_init
                modelprms['kp'] = kp_init
                modelprms['ddfsnow'] = ddfsnow_init
                modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                
                # Load sims df
                sims_fp = pygem_prms.emulator_fp + 'sims/' + glacier_str.split('.')[0].zfill(2) + '/'
                sims_fn = glacier_str + '-' + str(pygem_prms.emulator_sims) + '_emulator_sims.csv'

                if not os.path.exists(sims_fp + sims_fn) or pygem_prms.overwrite_em_sims:
                    # ----- Temperature bias bounds (ensure reasonable values) -----
                    # Tbias lower bound based on some bins having negative climatic mass balance
                    tbias_maxacc = (-1 * (gdir.historical_climate['temp'] + gdir.historical_climate['lr'] *
                                    (fls[0].surface_h.min() - gdir.historical_climate['elev'])).max())
                    modelprms['tbias'] = tbias_maxacc
                    nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                return_tbias_mustmelt_wmb=True)
                    while nbinyears_negmbclim < 10 or mb_mwea > mb_obs_mwea:
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                    return_tbias_mustmelt_wmb=True)
                        if debug:
                            print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3),
                                  'nbinyears_negmbclim:', nbinyears_negmbclim)        
                    tbias_stepsmall = 0.05
                    while nbinyears_negmbclim > 10:
                        modelprms['tbias'] = modelprms['tbias'] - tbias_stepsmall
                        nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                    return_tbias_mustmelt_wmb=True)
                        if debug:
                            print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3),
                                  'nbinyears_negmbclim:', nbinyears_negmbclim)
                    # Tbias lower bound 
                    tbias_bndlow = modelprms['tbias'] + tbias_stepsmall
                    modelprms['tbias'] = tbias_bndlow
                    nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                return_tbias_mustmelt_wmb=True)
                    output_all = np.array([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], 
                                          mb_mwea, nbinyears_negmbclim])
                    
                    # Tbias lower bound & high precipitation factor
                    modelprms['kp'] = stats.gamma.ppf(0.99, pygem_prms.kp_gamma_alpha, scale=1/pygem_prms.kp_gamma_beta)
                    nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                return_tbias_mustmelt_wmb=True)
                    output_single = np.array([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], 
                                              mb_mwea, nbinyears_negmbclim])
                    output_all = np.vstack((output_all, output_single))
                    
                    if debug:
                        print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                              'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))

                    # Tbias 'mid-point'
                    modelprms['kp'] = pygem_prms.kp_init
                    ncount_tbias = 0
                    tbias_bndhigh = 10
                    tbias_middle = tbias_bndlow + tbias_step
                    while mb_mwea > mb_obs_mwea and modelprms['tbias'] < 50:
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                    return_tbias_mustmelt_wmb=True)
                        output_single = np.array([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], 
                                                  mb_mwea, nbinyears_negmbclim])
                        output_all = np.vstack((output_all, output_single))
                        tbias_middle = modelprms['tbias'] - tbias_step / 2
                        ncount_tbias += 1
                        if debug:
                            print(ncount_tbias, 
                                  'tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))
                    
                    # Tbias upper bound (run for equal amount of steps above the midpoint)
                    while ncount_tbias > 0:
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                    return_tbias_mustmelt_wmb=True)
                        output_single = np.array([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], 
                                                  mb_mwea, nbinyears_negmbclim])
                        output_all = np.vstack((output_all, output_single))
                        tbias_bndhigh = modelprms['tbias']
                        ncount_tbias -= 1
                        if debug:
                            print(ncount_tbias, 
                                  'tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))
    
                    # ------ RANDOM RUNS -------
                    # Temperature bias
                    if pygem_prms.tbias_disttype == 'uniform':
                        tbias_random = np.random.uniform(low=tbias_bndlow, high=tbias_bndhigh, 
                                                        size=pygem_prms.emulator_sims)
                    elif pygem_prms.tbias_disttype == 'truncnormal':
                        tbias_zlow = (tbias_bndlow - tbias_middle) / pygem_prms.tbias_sigma
                        tbias_zhigh = (tbias_bndhigh - tbias_middle) / pygem_prms.tbias_sigma
                        tbias_random = stats.truncnorm.rvs(a=tbias_zlow, b=tbias_zhigh, loc=tbias_middle,
                                                          scale=pygem_prms.tbias_sigma, size=pygem_prms.emulator_sims)
                    if debug:
                        print('\ntbias random:', tbias_random.mean(), tbias_random.std())
                    
                    # Precipitation factor
                    kp_random = stats.gamma.rvs(pygem_prms.kp_gamma_alpha, scale=1/pygem_prms.kp_gamma_beta, 
                                                size=pygem_prms.emulator_sims)
                    if debug:
                        print('kp random:', kp_random.mean(), kp_random.std())
                    
                    # Degree-day factor of snow
                    ddfsnow_zlow = (pygem_prms.ddfsnow_bndlow - pygem_prms.ddfsnow_mu) / pygem_prms.ddfsnow_sigma
                    ddfsnow_zhigh = (pygem_prms.ddfsnow_bndhigh - pygem_prms.ddfsnow_mu) / pygem_prms.ddfsnow_sigma
                    ddfsnow_random = stats.truncnorm.rvs(a=ddfsnow_zlow, b=ddfsnow_zhigh, loc=pygem_prms.ddfsnow_mu,
                                                        scale=pygem_prms.ddfsnow_sigma, size=pygem_prms.emulator_sims)
                    if debug:    
                        print('ddfsnow random:', ddfsnow_random.mean(), ddfsnow_random.std(),'\n')
                    
                    # Run through random values
                    for nsim in range(pygem_prms.emulator_sims):
                        modelprms['tbias'] = tbias_random[nsim]
                        modelprms['kp'] = kp_random[nsim]
                        modelprms['ddfsnow'] = ddfsnow_random[nsim]
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                        nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                    return_tbias_mustmelt_wmb=True)
                        
                        
                        output_single = np.array([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], 
                                                  mb_mwea, nbinyears_negmbclim])
                        output_all = np.vstack((output_all, output_single))
                        if debug and nsim%500 == 0:
                            print(nsim, 'tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))
                    
                    # ----- Export results -----
                    sims_df = pd.DataFrame(output_all, columns=['tbias', 'kp', 'ddfsnow', 'mb_mwea', 
                                                                'nbinyrs_negmbclim'])
                    if os.path.exists(sims_fp) == False:
                        os.makedirs(sims_fp, exist_ok=True)
                    sims_df.to_csv(sims_fp + sims_fn, index=False)
                
                else:
                    # Load simulations
                    sims_df = pd.read_csv(sims_fp + sims_fn)

                # ----- EMULATOR: Mass balance -----
                em_mod_fn = glacier_str + '-emulator-mb_mwea.pth'
                em_mod_fp = pygem_prms.emulator_fp + 'models/' + glacier_str.split('.')[0].zfill(2) + '/'
                if not os.path.exists(em_mod_fp + em_mod_fn)  or pygem_prms.overwrite_em_sims:
                    mbEmulator = create_emulator(glacier_str, sims_df, y_cn='mb_mwea', debug=debug)
                else:
                    mbEmulator = massbalEmulator.load(em_mod_path = em_mod_fp + em_mod_fn)

                # ===== HH2015 MODIFIED CALIBRATION USING EMULATOR =====
                if pygem_prms.opt_hh2015_mod:
                    tbias_init = pygem_prms.tbias_init
                    tbias_step = pygem_prms.tbias_step
                    kp_init = pygem_prms.kp_init
                    kp_bndlow = pygem_prms.kp_bndlow
                    kp_bndhigh = pygem_prms.kp_bndhigh
                    ddfsnow_init = pygem_prms.ddfsnow_init

                    # ----- FUNCTIONS: COMPUTATIONALLY FASTER AND MORE ROBUST THAN SCIPY MINIMIZE -----
                    def update_bnds(prm2opt, prm_bndlow, prm_bndhigh, prm_mid, mb_mwea_low, mb_mwea_high, mb_mwea_mid,
                                    debug=False):
                        """ Update bounds for various parameters for the single_param_optimizer """
                        # If mass balance less than observation, reduce tbias
                        if prm2opt == 'kp':
                            if mb_mwea_mid < mb_obs_mwea:
                                prm_bndlow_new, mb_mwea_low_new = prm_mid, mb_mwea_mid
                                prm_bndhigh_new, mb_mwea_high_new = prm_bndhigh, mb_mwea_high
                            else:
                                prm_bndlow_new, mb_mwea_low_new = prm_bndlow, mb_mwea_low
                                prm_bndhigh_new, mb_mwea_high_new = prm_mid, mb_mwea_mid
                        elif prm2opt == 'ddfsnow':
                            if mb_mwea_mid < mb_obs_mwea:
                                prm_bndlow_new, mb_mwea_low_new = prm_bndlow, mb_mwea_low
                                prm_bndhigh_new, mb_mwea_high_new = prm_mid, mb_mwea_mid
                            else:
                                prm_bndlow_new, mb_mwea_low_new = prm_mid, mb_mwea_mid
                                prm_bndhigh_new, mb_mwea_high_new = prm_bndhigh, mb_mwea_high
                        elif prm2opt == 'tbias':
                            if mb_mwea_mid < mb_obs_mwea:
                                prm_bndlow_new, mb_mwea_low_new = prm_bndlow, mb_mwea_low
                                prm_bndhigh_new, mb_mwea_high_new = prm_mid, mb_mwea_mid
                            else:
                                prm_bndlow_new, mb_mwea_low_new = prm_mid, mb_mwea_mid
                                prm_bndhigh_new, mb_mwea_high_new = prm_bndhigh, mb_mwea_high
                        
                        prm_mid_new = (prm_bndlow_new + prm_bndhigh_new) / 2
                        modelprms[prm2opt] = prm_mid_new
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                        mb_mwea_mid_new = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
    
                        if debug:
                            print(prm2opt + '_bndlow:', np.round(prm_bndlow_new,2), 
                                  'mb_mwea_low:', np.round(mb_mwea_low_new,2))                        
                            print(prm2opt + '_bndhigh:', np.round(prm_bndhigh_new,2), 
                                  'mb_mwea_high:', np.round(mb_mwea_high_new,2))                        
                            print(prm2opt + '_mid:', np.round(prm_mid_new,2), 
                                  'mb_mwea_mid:', np.round(mb_mwea_mid_new,3))
                        
                        return (prm_bndlow_new, prm_bndhigh_new, prm_mid_new, 
                                mb_mwea_low_new, mb_mwea_high_new, mb_mwea_mid_new)
                        
                        
                    def single_param_optimizer(modelprms_subset, mb_obs_mwea, prm2opt=None,
                                              kp_bnds=None, tbias_bnds=None, ddfsnow_bnds=None,
                                              mb_mwea_threshold=0.005, debug=False):
                        """ Single parameter optimizer based on a mid-point approach 
                        
                        Computationally more robust and sometimes faster than scipy minimize
                        """
                        assert prm2opt is not None, 'For single_param_optimizer you must specify parameter to optimize'
                  
                        if prm2opt == 'kp':
                            prm_bndlow = kp_bnds[0]
                            prm_bndhigh = kp_bnds[1]
                            modelprms['tbias'] = modelprms_subset['tbias']
                            modelprms['ddfsnow'] = modelprms_subset['ddfsnow']
                        elif prm2opt == 'ddfsnow':
                            prm_bndlow = ddfsnow_bnds[0]
                            prm_bndhigh = ddfsnow_bnds[1]
                            modelprms['kp'] = modelprms_subset['kp']
                            modelprms['tbias'] = modelprms_subset['tbias']
                        elif prm2opt == 'tbias':
                            prm_bndlow = tbias_bnds[0]
                            prm_bndhigh = tbias_bnds[1]
                            modelprms['kp'] = modelprms_subset['kp']
                            modelprms['ddfsnow'] = modelprms_subset['ddfsnow']
    
                        # Lower bound
                        modelprms[prm2opt] = prm_bndlow
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                        mb_mwea_low = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                        # Upper bound
                        modelprms[prm2opt] = prm_bndhigh
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                        mb_mwea_high = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                        # Middle bound
                        prm_mid = (prm_bndlow + prm_bndhigh) / 2
                        modelprms[prm2opt] = prm_mid
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                        mb_mwea_mid = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                        
                        if debug:
                            print(prm2opt + '_bndlow:', np.round(prm_bndlow,2), 'mb_mwea_low:', np.round(mb_mwea_low,2))
                            print(prm2opt + '_bndhigh:', np.round(prm_bndhigh,2), 'mb_mwea_high:', np.round(mb_mwea_high,2))
                            print(prm2opt + '_mid:', np.round(prm_mid,2), 'mb_mwea_mid:', np.round(mb_mwea_mid,3))
                        
                        # Optimize the model parameter
                        if np.absolute(mb_mwea_low - mb_obs_mwea) <= mb_mwea_threshold:
                            modelprms[prm2opt] = prm_bndlow
                            mb_mwea_mid = mb_mwea_low
                        elif np.absolute(mb_mwea_low - mb_obs_mwea) <= mb_mwea_threshold:
                            modelprms[prm2opt] = prm_bndhigh
                            mb_mwea_mid = mb_mwea_high
                        else:
                            ncount = 0
                            while (np.absolute(mb_mwea_mid - mb_obs_mwea) > mb_mwea_threshold and 
                                  np.absolute(mb_mwea_low - mb_mwea_high) > mb_mwea_threshold):
                                if debug:
                                    print('\n ncount:', ncount)
                                (prm_bndlow, prm_bndhigh, prm_mid, mb_mwea_low, mb_mwea_high, mb_mwea_mid) = (
                                        update_bnds(prm2opt, prm_bndlow, prm_bndhigh, prm_mid, 
                                                    mb_mwea_low, mb_mwea_high, mb_mwea_mid, debug=debug))
                                ncount += 1
                            
                        return modelprms, mb_mwea_mid


                    # ===== SET THINGS UP ======
                    if debug:
                        sims_df['mb_em'] = np.nan
                        for nidx in sims_df.index.values:
                            modelprms['tbias'] = sims_df.loc[nidx,'tbias']
                            modelprms['kp'] = sims_df.loc[nidx,'kp']
                            modelprms['ddfsnow'] = sims_df.loc[nidx,'ddfsnow']
                            sims_df.loc[nidx,'mb_em'] = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                        sims_df['mb_em_dif'] = sims_df['mb_em'] - sims_df['mb_mwea'] 
                    
                    # ----- TEMPERATURE BIAS BOUNDS -----
                    # Selects from emulator sims dataframe
                    sims_df_subset = sims_df.loc[sims_df['kp']==1, :]
                    tbias_bndhigh = float(sims_df_subset['tbias'].max())
                    tbias_bndlow = float(sims_df_subset['tbias'].min())
                    
                    # Adjust tbias_init based on bounds
                    if tbias_init > tbias_bndhigh:
                        tbias_init = tbias_bndhigh
                    elif tbias_init < tbias_bndlow:
                        tbias_init = tbias_bndlow
                        
                    # ----- Mass balance bounds -----
                    # Upper bound
                    modelprms['kp'] = kp_bndhigh
                    modelprms['tbias'] = tbias_bndlow
                    modelprms['ddfsnow'] = ddfsnow_init
                    mb_mwea_bndhigh = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                    # Lower bound
                    modelprms['kp'] = kp_bndlow
                    modelprms['tbias'] = tbias_bndhigh
                    modelprms['ddfsnow'] = ddfsnow_init
                    mb_mwea_bndlow = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                    if debug:
                        print('mb_mwea_max:', np.round(mb_mwea_bndhigh,2),
                              'mb_mwea_min:', np.round(mb_mwea_bndlow,2))

                    if mb_obs_mwea > mb_mwea_bndhigh:
                        continue_param_search = False
                        tbias_opt = tbias_bndlow
                        kp_opt= kp_bndhigh
                        troubleshoot_fp = (pygem_prms.output_filepath + 'errors/' + 
                                          pygem_prms.option_calibration + '/' + 
                                          glacier_str.split('.')[0].zfill(2) + '/')
                        if not os.path.exists(troubleshoot_fp):
                            os.makedirs(troubleshoot_fp, exist_ok=True)
                        txt_fn_extrapfail = glacier_str + "-mbs_obs_outside_bnds.txt"
                        with open(troubleshoot_fp + txt_fn_extrapfail, "w") as text_file:
                            text_file.write(glacier_str + ' observed mass balance exceeds max accumulation ' +
                                            'with value of ' + str(np.round(mb_obs_mwea,2)) + ' mwea')
                        
                    elif mb_obs_mwea < mb_mwea_bndlow:
                        continue_param_search = False
                        tbias_opt = tbias_bndhigh
                        kp_opt= kp_bndlow
                        troubleshoot_fp = (pygem_prms.output_filepath + 'errors/' + 
                                          pygem_prms.option_calibration + '/' + 
                                          glacier_str.split('.')[0].zfill(2) + '/')
                        if not os.path.exists(troubleshoot_fp):
                            os.makedirs(troubleshoot_fp, exist_ok=True)
                        txt_fn_extrapfail = glacier_str + "-mbs_obs_outside_bnds.txt"
                        with open(troubleshoot_fp + txt_fn_extrapfail, "w") as text_file:
                            text_file.write(glacier_str + ' observed mass balance below max loss ' +
                                            'with value of ' + str(np.round(mb_obs_mwea,2)) + ' mwea')
                    else:
                        continue_param_search = True
                        
                        # ===== ADJUST LOWER AND UPPER BOUNDS TO SET UP OPTIMIZATION ======
                        # Initialize model parameters
                        modelprms['tbias'] = tbias_init
                        modelprms['kp'] = kp_init
                        modelprms['ddfsnow'] = ddfsnow_init
                        
                        test_count = 0
                        test_count_acc = 0
                        mb_mwea = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                        if mb_mwea > mb_obs_mwea:
                            if debug:
                                print('increase tbias, decrease kp')
                            kp_bndhigh = 1
                            # Check if lower bound causes good agreement
                            modelprms['kp'] = kp_bndlow
                            mb_mwea = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                            if debug:
                                    print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                          'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))
                            while mb_mwea > mb_obs_mwea and test_count < 100:
                                # Update temperature bias
                                modelprms['tbias'] = modelprms['tbias'] + tbias_step
                                # Update bounds
                                tbias_bndhigh_opt = modelprms['tbias']
                                tbias_bndlow_opt = modelprms['tbias'] - tbias_step
                                # Compute mass balance
                                mb_mwea = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                                if debug:
                                    print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                          'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))
                                test_count += 1
                        else:
                            if debug:
                                print('decrease tbias, increase kp')
                            kp_bndlow = 1
                            # Check if upper bound causes good agreement
                            modelprms['kp'] = kp_bndhigh
                            mb_mwea = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                            if debug:
                                print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                      'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))
                            while mb_obs_mwea > mb_mwea and test_count < 100:
                                # Update temperature bias
                                modelprms['tbias'] = modelprms['tbias'] - tbias_step
                                # If temperature bias is at lower limit, then increase precipitation factor
                                if modelprms['tbias'] <= tbias_bndlow:
                                    modelprms['tbias'] = tbias_bndlow
                                    if test_count_acc > 0:
                                        kp_bndhigh = kp_bndhigh + 1
                                        modelprms['kp'] = kp_bndhigh
                                    test_count_acc += 1
                                # Update bounds (must do after potential correction for lower bound)
                                tbias_bndlow_opt = modelprms['tbias']
                                tbias_bndhigh_opt = modelprms['tbias'] + tbias_step
                                # Compute mass balance
                                mb_mwea = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                                if debug:
                                    print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                          'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))
                                test_count += 1

                        # ===== ROUND 1: PRECIPITATION FACTOR ======    
                        if debug:
                            print('Round 1:')
                            print(glacier_str + '  kp: ' + str(np.round(modelprms['kp'],2)) +
                                  ' ddfsnow: ' + str(np.round(modelprms['ddfsnow'],4)) +
                                  ' tbias: ' + str(np.round(modelprms['tbias'],2)))
                        
                        # Reset parameters
                        modelprms['tbias'] = tbias_init
                        modelprms['kp'] = kp_init
                        modelprms['ddfsnow'] = ddfsnow_init
                        
                        # Lower bound
                        modelprms['kp'] = kp_bndlow
                        mb_mwea_kp_low = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                        # Upper bound
                        modelprms['kp'] = kp_bndhigh
                        mb_mwea_kp_high = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                        
                        # Optimal precipitation factor
                        if mb_obs_mwea < mb_mwea_kp_low:
                            kp_opt = kp_bndlow
                            mb_mwea = mb_mwea_kp_low
                        elif mb_obs_mwea > mb_mwea_kp_high:
                            kp_opt = kp_bndhigh
                            mb_mwea = mb_mwea_kp_high
                        else:
                            # Single parameter optimizer (computationally more efficient and less prone to fail)
                            modelprms_subset = {'kp':kp_init, 'ddfsnow': ddfsnow_init, 'tbias': tbias_init}
                            kp_bnds = (kp_bndlow, kp_bndhigh)
                            modelprms_opt, mb_mwea = single_param_optimizer(
                                    modelprms_subset, mb_obs_mwea, prm2opt='kp', kp_bnds=kp_bnds, debug=debug)
                            kp_opt = modelprms_opt['kp']
                            continue_param_search = False
                            
                        # Update parameter values
                        modelprms['kp'] = kp_opt
                        if debug:
                            print('  kp:', np.round(kp_opt,2), 'mb_mwea:', np.round(mb_mwea,3), 
                                  'obs_mwea:', np.round(mb_obs_mwea,3))

                        # ===== ROUND 2: TEMPERATURE BIAS ======
                        if continue_param_search:
                            if debug:
                                print('Round 2:')    
                            # Single parameter optimizer (computationally more efficient and less prone to fail)
                            modelprms_subset = {'kp':kp_opt, 'ddfsnow': ddfsnow_init, 
                                                'tbias': np.mean([tbias_bndlow_opt, tbias_bndhigh_opt])}
                            tbias_bnds = (tbias_bndlow_opt, tbias_bndhigh_opt)                            
                            modelprms_opt, mb_mwea = single_param_optimizer(
                                    modelprms_subset, mb_obs_mwea, prm2opt='tbias', tbias_bnds=tbias_bnds, debug=debug)
                            
                            # Update parameter values
                            tbias_opt = modelprms_opt['tbias']
                            modelprms['tbias'] = tbias_opt
                            if debug:
                                print('  tbias:', np.round(tbias_opt,3), 'mb_mwea:', np.round(mb_mwea,3), 
                                      'obs_mwea:', np.round(mb_obs_mwea,3))
        
                        else:
                            tbias_opt = modelprms['tbias']
                        
                        
                        if debug:
                            print('\n\ntbias:', np.round(tbias_opt,2), 'kp:', np.round(kp_opt,2),
                                  'mb_mwea:', np.round(mb_mwea,3), 'obs_mwea:', np.round(mb_obs_mwea,3), '\n\n')
                    
                    modelparams_opt = modelprms
                    modelparams_opt['kp'] = kp_opt
                    modelparams_opt['tbias'] = tbias_opt
                    
                    # Export model parameters
                    modelprms = modelparams_opt
                    for vn in ['ddfice', 'ddfsnow', 'kp', 'precgrad', 'tbias', 'tsnow_threshold']:
                        modelprms[vn] = [modelprms[vn]]
                    modelprms['mb_mwea'] = [float(mb_mwea)]
                    modelprms['mb_obs_mwea'] = [float(mb_obs_mwea)]
                    modelprms['mb_obs_mwea_err'] = [float(mb_obs_mwea_err)]
                    
                    modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                    modelprms_fp = (pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                    + '/')
                    if not os.path.exists(modelprms_fp):
                        os.makedirs(modelprms_fp, exist_ok=True)
                    modelprms_fullfn = modelprms_fp + modelprms_fn
                    if os.path.exists(modelprms_fullfn):
                        with open(modelprms_fullfn, 'rb') as f:
                            modelprms_dict = pickle.load(f)
                        modelprms_dict[pygem_prms.option_calibration] = modelprms
                    else:
                        modelprms_dict = {pygem_prms.option_calibration: modelprms}
                    with open(modelprms_fullfn, 'wb') as f:
                        pickle.dump(modelprms_dict, f)
            
            #%% ===== MCMC CALIBRATION ======
            # use MCMC method to determine posterior probability distributions of the three parameters tbias,
            # ddfsnow and kp. Then create an ensemble of parameter sets evenly sampled from these
            # distributions, and output these sets of parameters and their corresponding mass balances to be
            # used in the simulations.
            elif pygem_prms.option_calibration == 'MCMC':
                if pygem_prms.option_use_emulator:
                    # load emulator
                    em_mod_fn = glacier_str + '-emulator-mb_mwea.pth'
                    em_mod_fp = pygem_prms.emulator_fp + 'models/' + glacier_str.split('.')[0].zfill(2) + '/'
                    assert os.path.exists(em_mod_fp + em_mod_fn), f'emulator output does not exist : {em_mod_fp + em_mod_fn}'
                    mbEmulator = massbalEmulator.load(em_mod_path = em_mod_fp + em_mod_fn)
                    outpath_sfix = ''               # output file path suffix if using emulator
                else:
                    outpath_sfix = '-fullsim'       # output file path suffix if not using emulator

                # ---------------------------------                    
                # ----- FUNCTION DECLARATIONS -----                    
                # ---------------------------------                    
                # Rough estimate of minimum elevation mass balance function
                def calc_mb_total_minelev(modelprms):
                    """ Approximate estimate of the mass balance at minimum elevation """
                    fl = fls[0]
                    min_elev = fl.surface_h.min()
                    glacier_gcm_temp = gdir.historical_climate['temp']
                    glacier_gcm_prec = gdir.historical_climate['prec']
                    glacier_gcm_lr = gdir.historical_climate['lr']
                    glacier_gcm_elev = gdir.historical_climate['elev']
                    # Temperature using gcm and glacier lapse rates
                    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tempchange
                    T_minelev = (glacier_gcm_temp + glacier_gcm_lr *
                                (glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale] - glacier_gcm_elev) +
                                glacier_gcm_lr * 
                                (min_elev - glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale]) +
                                modelprms['tbias'])
                    # Precipitation using precipitation factor and precipitation gradient
                    #  P_bin = P_gcm * prec_factor * (1 + prec_grad * (z_bin - z_ref))
                    P_minelev = (glacier_gcm_prec * modelprms['kp'] * (1 + modelprms['precgrad'] * (min_elev -
                                glacier_rgi_table.loc[pygem_prms.option_elev_ref_downscale])))
                    # Accumulation using tsnow_threshold
                    Acc_minelev = np.zeros(P_minelev.shape)
                    Acc_minelev[T_minelev <= modelprms['tsnow_threshold']] = (
                            P_minelev[T_minelev <= modelprms['tsnow_threshold']])
                    # Melt
                    # energy available for melt [degC day]
                    melt_energy_available = T_minelev * dates_table['daysinmonth'].values
                    melt_energy_available[melt_energy_available < 0] = 0
                    # assume all snow melt because anything more would melt underlying ice in lowermost bin
                    # SNOW MELT [m w.e.]
                    Melt_minelev = modelprms['ddfsnow'] * melt_energy_available
                    # Total mass balance over entire period at minimum elvation
                    mb_total_minelev = (Acc_minelev - Melt_minelev).sum()
                
                    return mb_total_minelev
                
                def mb_max(*args, **kwargs):
                    """ Model parameters cannot completely melt the glacier (psuedo-likelihood fxn) """
                    if kwargs['massbal'] < mb_max_loss:
                        return -np.inf
                    else:
                        return 0

                def must_melt(kp, tbias, ddfsnow, **kwargs):
                    """ Likelihood function for mass balance [mwea] based on model parametersr (psuedo-likelihood fxn) """                          
                    modelprms_copy = modelprms.copy()
                    modelprms_copy['tbias'] = float(tbias)
                    modelprms_copy['kp'] = float(kp)
                    modelprms_copy['ddfsnow'] = float(ddfsnow)
                    modelprms_copy['ddfice'] = modelprms_copy['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_total_minelev = calc_mb_total_minelev(modelprms_copy)
                    if mb_total_minelev < 0:
                        return 0
                    else:
                        return -np.inf
                # ---------------------------------                    

                # ---------------------------------
                # ----- MASS BALANCE MAX LOSS -----
                # ---------------------------------
                # Maximum mass loss [mwea] (based on consensus ice thickness estimate)
                # consensus_mass has units of kg
                if os.path.exists(gdir.get_filepath('consensus_mass')):
                    with open(gdir.get_filepath('consensus_mass'), 'rb') as f:
                        consensus_mass = pickle.load(f)
                else:
                    # Mean global ice thickness from Farinotti et al. (2019) used for missing consensus glaciers
                    ice_thickness_constant = 224
                    consensus_mass = glacier_rgi_table.Area * 1e6 * ice_thickness_constant * pygem_prms.density_ice

                mb_max_loss = (-1 * consensus_mass / pygem_prms.density_water / gdir.rgi_area_m2 / 
                                (gdir.dates_table.shape[0] / 12))
                # ---------------------------------

                # ------------------
                # ----- PRIORS -----
                # ------------------
                # Prior distributions (specified or informed by regions)
                if pygem_prms.priors_reg_fullfn is not None:
                    # Load priors
                    priors_df = pd.read_csv(pygem_prms.priors_reg_fullfn)
                    priors_idx = np.where((priors_df.O1Region == glacier_rgi_table['O1Region']) & 
                                            (priors_df.O2Region == glacier_rgi_table['O2Region']))[0][0]
                    # Precipitation factor priors
                    kp_gamma_alpha = float(priors_df.loc[priors_idx, 'kp_alpha'])
                    kp_gamma_beta = float(priors_df.loc[priors_idx, 'kp_beta'])
                    # Temperature bias priors
                    tbias_mu = float(priors_df.loc[priors_idx, 'tbias_mean'])
                    tbias_sigma = float(priors_df.loc[priors_idx, 'tbias_std'])
                else:
                    # Precipitation factor priors
                    kp_gamma_alpha = pygem_prms.kp_gamma_alpha
                    kp_gamma_beta = pygem_prms.kp_gamma_beta
                    # Temperature bias priors
                    tbias_mu = pygem_prms.tbias_mu
                    tbias_sigma = pygem_prms.tbias_sigma

                # put all priors info together into a dictionary
                priors =    {
                            'tbias':    {'type':'normal', 'mu':float(tbias_mu) , 'sigma':float(tbias_sigma)},
                            'kp':      {'type':'gamma', 'alpha':float(kp_gamma_alpha), 'beta':float(kp_gamma_beta)},
                            'ddfsnow':  {'type':'truncated_normal', 'mu':pygem_prms.ddfsnow_mu, 'sigma':pygem_prms.ddfsnow_sigma ,'a':pygem_prms.ddfsnow_bndlow, 'b':pygem_prms.ddfsnow_bndhigh },
                            }
                if priors['kp']['type'] == 'gamma':
                    priors['kp']['mu'] = priors['kp']['alpha'] / priors['kp']['beta']
                    priors['kp']['sigma'] = math.sqrt(priors['kp']['alpha']) / priors['kp']['beta']
                # ------------------

                # -----------------------------------
                # ----- TEMPERATURE BIAS BOUNDS -----
                # -----------------------------------
                # note, temperature bias bounds will remain constant across chains if using emulator
                if pygem_prms.option_use_emulator:
                    # Selects from emulator sims dataframe
                    sims_fp = pygem_prms.emulator_fp + 'sims/' + glacier_str.split('.')[0].zfill(2) + '/'
                    sims_fn = glacier_str + '-' + str(pygem_prms.emulator_sims) + '_emulator_sims.csv'
                    sims_df = pd.read_csv(sims_fp + sims_fn)
                    sims_df_subset = sims_df.loc[sims_df['kp']==1, :]
                    tbias_bndhigh = float(sims_df_subset['tbias'].max())
                    tbias_bndlow = float(sims_df_subset['tbias'].min())
                # -----------------------------------

                # prepare export modelprms dictionary
                modelprms_export = {}
                for k in ['tbias','kp','ddfsnow','ddfice','mb_mwea','ar']:
                    modelprms_export[k] = {}

                # ===== RUNNING MCMC =====
                try:
                # for batman in [0]:

                    ### loop over chains, adjust initial guesses accordingly ###
                    for n_chain in range(0,pygem_prms.n_chains):
    
                        if debug:
                            print('\n', glacier_str, ' chain ' + str(n_chain))

                        if n_chain == 0:
                            # Starting values: middle
                            tbias_start = tbias_mu
                            kp_start = kp_gamma_alpha / kp_gamma_beta
                            ddfsnow_start = pygem_prms.ddfsnow_mu

                        elif n_chain == 1:
                            # Starting values: lowest
                            tbias_start = tbias_mu - 1.96 * tbias_sigma
                            ddfsnow_start = pygem_prms.ddfsnow_mu - 1.96 * pygem_prms.ddfsnow_sigma
                            kp_start = stats.gamma.ppf(0.05,kp_gamma_alpha, scale=1/kp_gamma_beta)

                        elif n_chain == 2:
                            # Starting values: high
                            tbias_start = tbias_mu + 1.96 * tbias_sigma
                            ddfsnow_start = pygem_prms.ddfsnow_mu + 1.96 * pygem_prms.ddfsnow_sigma
                            kp_start = stats.gamma.ppf(0.95,kp_gamma_alpha, scale=1/kp_gamma_beta)

                        # store starting values in modelprms dictionary - tbias may change based on lower and upper bounds
                        modelprms['kp'] = kp_start
                        modelprms['ddfsnow'] = ddfsnow_start
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio

                        # -----------------------------------
                        # ----- TEMPERATURE BIAS BOUNDS -----
                        # -----------------------------------
                        # note, temperature bias bounds change between chains (below we establish reasonable lower and upper bounds to adjust the starting value if necessary)
                        if not pygem_prms.option_use_emulator:
                            # Determine bounds to check TC starting values and estimate maximum mass loss
                            modelprms['kp'] = kp_start
                            modelprms['ddfsnow'] = ddfsnow_start
                            modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                            # Tbias lower bound based on some bins having negative climatic mass balance
                            modelprms['tbias'] = (-1 * (gdir.historical_climate['temp'] + gdir.historical_climate['lr'] *
                                            (fls[0].surface_h.min() - gdir.historical_climate['elev'])).max()).item()
                            nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                        return_tbias_mustmelt_wmb=True)
                            while nbinyears_negmbclim < 10 or mb_mwea > mb_obs_mwea:
                                modelprms['tbias'] = modelprms['tbias'] + pygem_prms.tbias_step
                                nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                            return_tbias_mustmelt_wmb=True)
                                if debug:
                                    print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                            'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3),
                                            'nbinyears_negmbclim:', nbinyears_negmbclim)        
                            while nbinyears_negmbclim > 10:
                                modelprms['tbias'] = modelprms['tbias'] - pygem_prms.tbias_stepsmall
                                nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                            return_tbias_mustmelt_wmb=True)
                                if debug:
                                    print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                            'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3),
                                            'nbinyears_negmbclim:', nbinyears_negmbclim)
                            # Tbias lower bound 
                            tbias_bndlow = modelprms['tbias'] + pygem_prms.tbias_stepsmall
                            modelprms['tbias'] = tbias_bndlow
                            nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                        return_tbias_mustmelt_wmb=True)

                            # Tbias lower bound & high precipitation factor
                            modelprms['kp'] = stats.gamma.ppf(0.99, kp_gamma_alpha, scale=1/kp_gamma_beta)
                            nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                        return_tbias_mustmelt_wmb=True)

                            if debug:
                                print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                      'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))
        
                            # Tbias 'mid-point'
                            modelprms['kp'] = 1
                            ncount_tbias = 0
                            tbias_bndhigh = 10
                            tbias_middle = tbias_bndlow + pygem_prms.tbias_step
                            while mb_mwea > mb_obs_mwea and modelprms['tbias'] < 50:
                                modelprms['tbias'] = modelprms['tbias'] + pygem_prms.tbias_step
                                nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                            return_tbias_mustmelt_wmb=True)

                                tbias_middle = modelprms['tbias'] - pygem_prms.tbias_step / 2
                                ncount_tbias += 1
                                if debug:
                                    print(ncount_tbias, 
                                          'tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                          'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))
                            
                            # Tbias upper bound (run for equal amount of steps above the midpoint)
                            while ncount_tbias > 0:
                                modelprms['tbias'] = modelprms['tbias'] + pygem_prms.tbias_step
                                nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                            return_tbias_mustmelt_wmb=True)

                                tbias_bndhigh = modelprms['tbias']
                                ncount_tbias -= 1
                                if debug:
                                    print(ncount_tbias, 
                                          'tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                          'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))
                        # -----------------------------------
             
                        if debug:
                            print('tbias_bndlow:', np.round(tbias_bndlow,2), 'tbias_bndhigh:', np.round(tbias_bndhigh,2))
                        
                        # Adjust tbias_init based on bounds
                        if tbias_start > tbias_bndhigh:
                            tbias_start = tbias_bndhigh
                        elif tbias_start < tbias_bndlow:
                            tbias_start = tbias_bndlow
                        
                        # update tbias accordingly
                        modelprms['tbias'] = tbias_start

                        # --------------------------------------------------------------
                        # ----- CHECK STARTING CONDITIONS (adjust tbias as needed) -----
                        # --------------------------------------------------------------
                        # get starting mass
                        if pygem_prms.option_use_emulator:
                            mb_mwea_start = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                        else:
                            mb_mwea_start = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)

                        # check that starting mass balance does not result in entire glacier melting over calibration period
                        while mb_mwea_start < mb_max_loss:
                            modelprms['tbias'] = modelprms['tbias'] - pygem_prms.tbias_step
                            if pygem_prms.option_use_emulator:
                                mb_mwea_start = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                            else:
                                mb_mwea_start = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)

                        # check that melting occurs at the lowest elevation bin
                        mb_total_minelev_start = calc_mb_total_minelev(modelprms)
                        while mb_total_minelev_start > 0 and mb_mwea_start > mb_max_loss:
                            modelprms['tbias'] = modelprms['tbias'] + pygem_prms.tbias_stepsmall
                            mb_total_minelev_start = calc_mb_total_minelev(modelprms)
                            if pygem_prms.option_use_emulator:
                                mb_mwea_start = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                            else:
                                mb_mwea_start = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        # --------------------------------------------------------------

                        if debug:
                            print(modelprms)

                        # -------------------
                        # --- set up MCMC ---
                        # -------------------
                        # mass balance observation and standard deviation
                        obs = [(torch.tensor([mb_obs_mwea]),torch.tensor([mb_obs_mwea_err]))]

                        # if running full model (no emulator), or calibrating against binned \delta h, several arguments are needed
                        if pygem_prms.option_calib_binned_dh:
                            mbfxn = get_binned_dh                                   # returns (mb_mwea, binned_dh)
                            mbargs = (gdir,                                         # arguments for get_binned_dh()
                                      modelprms, 
                                      glacier_rgi_table, 
                                      fls, 
                                      glen_a_multiplier, 
                                      fs, 
                                      pygem_inds, 
                                      gdir.deltah['bin_edges'])
                            # append deltah obs and undto obs list
                            # obs.append((torch.tensor(gdir.deltah['dh']),torch.tensor(gdir.deltah['sigma'])))
                            obs.append((torch.tensor(gdir.deltah['dh']),torch.tensor([10])))
                        elif pygem_prms.option_use_emulator:
                            mbfxn = mbEmulator.eval                                 # returns (mb_mwea)
                            mbargs = None                                           # no additional arguments for mbEmulator.eval()
                        else:
                            mbfxn = mb_mwea_calc                                    # returns (mb_mwea)
                            mbargs = (gdir, modelprms, glacier_rgi_table, fls)      # arguments for mb_mwea_calc()

                        # instantiate mbPosterior given priors, and observed values
                        # note, mbEmulator.eval expects the modelprms to be ordered like so: [tbias, kp, ddfsnow], so priors and initial guesses must also be ordered as such)
                        priors = {key: priors[key] for key in ['tbias','kp','ddfsnow'] if key in priors}
                        mb = mcmc.mbPosterior(obs, priors, mb_func=mbfxn, mb_args=mbargs, potential_fxns=[mb_max, must_melt])

                        # compile initial guesses and standardize by standard deviations
                        initial_guesses = torch.tensor([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']]).flatten()
                        initial_guesses_z = mcmc.z_normalize(initial_guesses, mb.means, mb.stds)

                        # instantiate sampler
                        sampler = mcmc.Metropolis(mb.means, mb.stds)

                        # draw samples
                        m_chain_z, pred_chain, m_primes_z, pred_primes, _, ar = sampler.sample(initial_guesses_z, 
                                                                                                    mb.log_posterior, 
                                                                                                    n_samples=pygem_prms.mcmc_sample_no, 
                                                                                                    h=pygem_prms.mcmc_step, 
                                                                                                    burnin=int(pygem_prms.mcmc_burn_pct/100*pygem_prms.mcmc_sample_no), 
                                                                                                    thin_factor=pygem_prms.thin_interval, 
                                                                                                    progress_bar=args.progress_bar)

                        # inverse z-normalize the samples to original parameter space
                        m_chain = mcmc.inverse_z_normalize(m_chain_z, mb.means, mb.stds)
                        m_primes = mcmc.inverse_z_normalize(m_primes_z, mb.means, mb.stds)

                        # concatenate mass balance
                        m_chain = torch.cat((m_chain, torch.tensor(pred_chain[0]).reshape(-1,1)), dim=1)
                        m_primes = torch.cat((m_primes, torch.tensor(pred_primes[0]).reshape(-1,1)), dim=1)

                        if debug:
                            # print('\nacceptance ratio:', model.step_method_dict[next(iter(model.stochastics))][0].ratio)
                            print('mb_mwea_mean:', np.round(torch.mean(m_chain[:,-1]).item(),3),
                                  'mb_mwea_std:', np.round(torch.std(m_chain[:,-1]).item(),3),
                                  '\nmb_obs_mean:', np.round(mb_obs_mwea,3), 'mb_obs_std:', np.round(mb_obs_mwea_err,3))
                            # plot chain
                            fp = (pygem_prms.output_filepath + f'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                    + '/fig/')
                            if pygem_prms.option_calib_binned_dh:
                                fp += 'dh/' 
                            os.makedirs(fp, exist_ok=True)
                            if args.num_simultaneous_processes > 1:
                                show=False
                            else:
                                show=True
                            mcmc.plot_chain(m_primes, m_chain, obs[0], ar, glacier_str, show=show, fpath=f'{fp}/{glacier_str}-chain{n_chain}.png')
                            for i in pred_chain.keys():
                                mcmc.plot_1t1(obs[i], pred_chain[i], glacier_str, show=show, fpath=f'{fp}/{glacier_str}-chain{n_chain}-1to1-{i}.png')

                        # Store data from model to be exported
                        chain_str = 'chain_' + str(n_chain)
                        modelprms_export['tbias'][chain_str] = m_chain[:,0].tolist()
                        modelprms_export['kp'][chain_str] = m_chain[:,1].tolist()
                        modelprms_export['ddfsnow'][chain_str] = m_chain[:,2].tolist()
                        modelprms_export['ddfice'][chain_str] = (m_chain[:,2] /
                                                                  pygem_prms.ddfsnow_iceratio).tolist()
                        modelprms_export['mb_mwea'][chain_str] = m_chain[:,3].tolist()
                        modelprms_export['ar'][chain_str] = ar
                        if pygem_prms.option_calib_binned_dh:
                            if 'dh' not in modelprms_export.keys():
                                modelprms_export['dh'] = {} # add key to export \delta h predictions
                            dh_preds = [preds.flatten().tolist() for preds in pred_chain[1]]
                            modelprms_export['dh'][chain_str] = dh_preds

                    # Export model parameters
                    modelprms_export['precgrad'] = [pygem_prms.precgrad]
                    modelprms_export['tsnow_threshold'] = [pygem_prms.tsnow_threshold]
                    modelprms_export['mb_obs_mwea'] = [float(mb_obs_mwea)]
                    modelprms_export['mb_obs_mwea_err'] = [float(mb_obs_mwea_err)]
                    modelprms_export['priors'] = priors
                    if pygem_prms.option_calib_binned_dh:
                        modelprms_export['dh']['x'] = ((gdir.deltah['bin_edges'][:-1] + gdir.deltah['bin_edges'][1:]) / 2).tolist()
                        modelprms_export['dh']['obs'] = [ob.flatten().tolist() for ob in obs[1]]
                        modelprms_export['dh']['date'] = gdir.deltah['timestamps']

                    modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                    modelprms_fp = [(pygem_prms.output_filepath + f'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                    + '/')]
                    if pygem_prms.option_calib_binned_dh:
                        modelprms_fp[0] += 'dh/'
                    # if not using emulator (running full model), save output in ./calibration/ and ./calibration-fullsim/
                    if not pygem_prms.option_use_emulator:
                        modelprms_fp.append(pygem_prms.output_filepath + f'calibration{outpath_sfix}/' + glacier_str.split('.')[0].zfill(2) 
                                        + '/')
                    for fp in modelprms_fp:
                        if not os.path.exists(fp):
                            os.makedirs(fp, exist_ok=True)
                        modelprms_fullfn = fp + modelprms_fn
                        if os.path.exists(modelprms_fullfn):
                            with open(modelprms_fullfn, 'rb') as f:
                                modelprms_dict = pickle.load(f)
                            modelprms_dict[pygem_prms.option_calibration] = modelprms_export
                        else:
                            modelprms_dict = {pygem_prms.option_calibration: modelprms_export}
                        with open(modelprms_fullfn, 'wb') as f:
                            pickle.dump(modelprms_dict, f)
                    
                    # MCMC LOG SUCCESS
                    mcmc_good_fp = pygem_prms.output_filepath + f'mcmc_success{outpath_sfix}/' + glacier_str.split('.')[0].zfill(2) + '/'
                    if not os.path.exists(mcmc_good_fp):
                        os.makedirs(mcmc_good_fp, exist_ok=True)
                    txt_fn_good = glacier_str + "-mcmc_success.txt"
                    with open(mcmc_good_fp + txt_fn_good, "w") as text_file:
                        text_file.write(glacier_str + ' successfully exported mcmc results')
                
                except Exception as err:
                    # MCMC LOG FAILURE
                    mcmc_fail_fp = pygem_prms.output_filepath + f'mcmc_fail{outpath_sfix}/' + glacier_str.split('.')[0].zfill(2) + '/'
                    if not os.path.exists(mcmc_fail_fp):
                        os.makedirs(mcmc_fail_fp, exist_ok=True)
                    txt_fn_fail = glacier_str + "-mcmc_fail.txt"
                    with open(mcmc_fail_fp + txt_fn_fail, "w") as text_file:
                        text_file.write(glacier_str + f' failed to complete MCMC: {err}')


            #%% ===== HUSS AND HOCK (2015) CALIBRATION =====
            elif pygem_prms.option_calibration == 'HH2015':
                tbias_init = pygem_prms.tbias_init
                tbias_step = pygem_prms.tbias_step
                kp_init = pygem_prms.kp_init
                kp_bndlow = pygem_prms.kp_bndlow
                kp_bndhigh = pygem_prms.kp_bndhigh
                ddfsnow_init = pygem_prms.ddfsnow_init
                ddfsnow_bndlow = pygem_prms.ddfsnow_bndlow
                ddfsnow_bndhigh = pygem_prms.ddfsnow_bndhigh

                # ----- Initialize model parameters -----
                modelprms['tbias'] = tbias_init
                modelprms['kp'] = kp_init
                modelprms['ddfsnow'] = ddfsnow_init
                modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                continue_param_search = True
                
                # ----- FUNCTIONS: COMPUTATIONALLY FASTER AND MORE ROBUST THAN SCIPY MINIMIZE -----
                def update_bnds(prm2opt, prm_bndlow, prm_bndhigh, prm_mid, mb_mwea_low, mb_mwea_high, mb_mwea_mid,
                                debug=False):
                    # If mass balance less than observation, reduce tbias
                    if prm2opt == 'kp':
                        if mb_mwea_mid < mb_obs_mwea:
                            prm_bndlow_new, mb_mwea_low_new = prm_mid, mb_mwea_mid
                            prm_bndhigh_new, mb_mwea_high_new = prm_bndhigh, mb_mwea_high
                        else:
                            prm_bndlow_new, mb_mwea_low_new = prm_bndlow, mb_mwea_low
                            prm_bndhigh_new, mb_mwea_high_new = prm_mid, mb_mwea_mid
                    elif prm2opt == 'ddfsnow':
                        if mb_mwea_mid < mb_obs_mwea:
                            prm_bndlow_new, mb_mwea_low_new = prm_bndlow, mb_mwea_low
                            prm_bndhigh_new, mb_mwea_high_new = prm_mid, mb_mwea_mid
                        else:
                            prm_bndlow_new, mb_mwea_low_new = prm_mid, mb_mwea_mid
                            prm_bndhigh_new, mb_mwea_high_new = prm_bndhigh, mb_mwea_high
                    elif prm2opt == 'tbias':
                        if mb_mwea_mid < mb_obs_mwea:
                            prm_bndlow_new, mb_mwea_low_new = prm_bndlow, mb_mwea_low
                            prm_bndhigh_new, mb_mwea_high_new = prm_mid, mb_mwea_mid
                        else:
                            prm_bndlow_new, mb_mwea_low_new = prm_mid, mb_mwea_mid
                            prm_bndhigh_new, mb_mwea_high_new = prm_bndhigh, mb_mwea_high
                    
                    prm_mid_new = (prm_bndlow_new + prm_bndhigh_new) / 2
                    modelprms[prm2opt] = prm_mid_new
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_mid_new = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)

                    if debug:
                        print(prm2opt + '_bndlow:', np.round(prm_bndlow_new,2), 
                              'mb_mwea_low:', np.round(mb_mwea_low_new,2))                        
                        print(prm2opt + '_bndhigh:', np.round(prm_bndhigh_new,2), 
                              'mb_mwea_high:', np.round(mb_mwea_high_new,2))                        
                        print(prm2opt + '_mid:', np.round(prm_mid_new,2), 
                              'mb_mwea_mid:', np.round(mb_mwea_mid_new,3))
                    
                    return (prm_bndlow_new, prm_bndhigh_new, prm_mid_new, 
                            mb_mwea_low_new, mb_mwea_high_new, mb_mwea_mid_new)
                    
                
                def single_param_optimizer(modelprms_subset, mb_obs_mwea, prm2opt=None,
                                          kp_bnds=None, tbias_bnds=None, ddfsnow_bnds=None,
                                          mb_mwea_threshold=0.005, debug=False):
                    assert prm2opt is not None, 'For single_param_optimizer you must specify parameter to optimize'
              
                    if prm2opt == 'kp':
                        prm_bndlow = kp_bnds[0]
                        prm_bndhigh = kp_bnds[1]
                        modelprms['tbias'] = modelprms_subset['tbias']
                        modelprms['ddfsnow'] = modelprms_subset['ddfsnow']
                    elif prm2opt == 'ddfsnow':
                        prm_bndlow = ddfsnow_bnds[0]
                        prm_bndhigh = ddfsnow_bnds[1]
                        modelprms['kp'] = modelprms_subset['kp']
                        modelprms['tbias'] = modelprms_subset['tbias']
                    elif prm2opt == 'tbias':
                        prm_bndlow = tbias_bnds[0]
                        prm_bndhigh = tbias_bnds[1]
                        modelprms['kp'] = modelprms_subset['kp']
                        modelprms['ddfsnow'] = modelprms_subset['ddfsnow']

                    # Lower bound
                    modelprms[prm2opt] = prm_bndlow
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_low = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Upper bound
                    modelprms[prm2opt] = prm_bndhigh
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_high = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Middle bound
                    prm_mid = (prm_bndlow + prm_bndhigh) / 2
                    modelprms[prm2opt] = prm_mid
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_mid = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    
                    if debug:
                        print(prm2opt + '_bndlow:', np.round(prm_bndlow,2), 'mb_mwea_low:', np.round(mb_mwea_low,2))
                        print(prm2opt + '_bndhigh:', np.round(prm_bndhigh,2), 'mb_mwea_high:', np.round(mb_mwea_high,2))
                        print(prm2opt + '_mid:', np.round(prm_mid,2), 'mb_mwea_mid:', np.round(mb_mwea_mid,3))
                    
                    # Optimize the model parameter
                    if np.absolute(mb_mwea_low - mb_obs_mwea) <= mb_mwea_threshold:
                        modelprms[prm2opt] = prm_bndlow
                        mb_mwea_mid = mb_mwea_low
                    elif np.absolute(mb_mwea_low - mb_obs_mwea) <= mb_mwea_threshold:
                        modelprms[prm2opt] = prm_bndhigh
                        mb_mwea_mid = mb_mwea_high
                    else:
                        ncount = 0
                        while np.absolute(mb_mwea_mid - mb_obs_mwea) > mb_mwea_threshold:
                            if debug:
                                print('\n ncount:', ncount)
                            (prm_bndlow, prm_bndhigh, prm_mid, mb_mwea_low, mb_mwea_high, mb_mwea_mid) = (
                                    update_bnds(prm2opt, prm_bndlow, prm_bndhigh, prm_mid, 
                                                mb_mwea_low, mb_mwea_high, mb_mwea_mid, debug=debug))
                            ncount += 1
                        
                    return modelprms, mb_mwea_mid
                
                
                # ===== ROUND 1: PRECIPITATION FACTOR ======
                if debug:
                    print('Round 1:')
                    
                if debug:
                    print(glacier_str + '  kp: ' + str(np.round(modelprms['kp'],2)) +
                          ' ddfsnow: ' + str(np.round(modelprms['ddfsnow'],4)) +
                          ' tbias: ' + str(np.round(modelprms['tbias'],2)))
                
                # Lower bound
                modelprms['kp'] = kp_bndlow
                mb_mwea_kp_low = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                # Upper bound
                modelprms['kp'] = kp_bndhigh
                mb_mwea_kp_high = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                
                # Optimal precipitation factor
                if mb_obs_mwea < mb_mwea_kp_low:
                    kp_opt = kp_bndlow
                    mb_mwea = mb_mwea_kp_low
                elif mb_obs_mwea > mb_mwea_kp_high:
                    kp_opt = kp_bndhigh
                    mb_mwea = mb_mwea_kp_high
                else:
                    # Single parameter optimizer (computationally more efficient and less prone to fail)
                    modelprms_subset = {'kp':kp_init, 'ddfsnow': ddfsnow_init, 'tbias': tbias_init}
                    kp_bnds = (kp_bndlow, kp_bndhigh)
                    modelprms_opt, mb_mwea = single_param_optimizer(
                            modelprms_subset, mb_obs_mwea, prm2opt='kp', kp_bnds=kp_bnds, debug=debug)
                    kp_opt = modelprms_opt['kp']
                    continue_param_search = False
                    
                # Update parameter values
                modelprms['kp'] = kp_opt
                if debug:
                    print('  kp:', np.round(kp_opt,2), 'mb_mwea:', np.round(mb_mwea,2))

                # ===== ROUND 2: DEGREE-DAY FACTOR OF SNOW ======
                if continue_param_search:
                    if debug:
                        print('Round 2:')
                    # Lower bound
                    modelprms['ddfsnow'] = ddfsnow_bndlow
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_ddflow = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Upper bound
                    modelprms['ddfsnow'] = ddfsnow_bndhigh
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    mb_mwea_ddfhigh = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Optimal degree-day factor of snow
                    if mb_obs_mwea < mb_mwea_ddfhigh:
                        ddfsnow_opt = ddfsnow_bndhigh
                        mb_mwea = mb_mwea_ddfhigh
                    elif mb_obs_mwea > mb_mwea_ddflow:
                        ddfsnow_opt = ddfsnow_bndlow
                        mb_mwea = mb_mwea_ddflow
                    else:
                        # Single parameter optimizer (computationally more efficient and less prone to fail)
                        modelprms_subset = {'kp':kp_opt, 'ddfsnow': ddfsnow_init, 'tbias': tbias_init}
                        ddfsnow_bnds = (ddfsnow_bndlow, ddfsnow_bndhigh)
                        modelprms_opt, mb_mwea = single_param_optimizer(
                                modelprms_subset, mb_obs_mwea, prm2opt='ddfsnow', ddfsnow_bnds=ddfsnow_bnds, debug=debug)
                        ddfsnow_opt = modelprms_opt['ddfsnow']
                        continue_param_search = False
                    # Update parameter values
                    modelprms['ddfsnow'] = ddfsnow_opt
                    modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                    if debug:
                        print('  ddfsnow:', np.round(ddfsnow_opt,4), 'mb_mwea:', np.round(mb_mwea,2))
                else:
                    ddfsnow_opt = modelprms['ddfsnow']
                        
                # ===== ROUND 3: TEMPERATURE BIAS ======
                if continue_param_search:
                    if debug:
                        print('Round 3:')
                    # ----- TEMPBIAS: max accumulation -----
                    # Lower temperature bound based on no positive temperatures
                    # Temperature at the lowest bin
                    #  T_bin = T_gcm + lr_gcm * (z_ref - z_gcm) + lr_glac * (z_bin - z_ref) + tbias
                    tbias_max_acc = (-1 * (gdir.historical_climate['temp'] + gdir.historical_climate['lr'] *
                                    (fls[0].surface_h.min() - gdir.historical_climate['elev'])).max())
                    tbias_bndlow = tbias_max_acc
                    modelprms['tbias'] = tbias_bndlow
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)

                    if debug:
                        print('  tbias_bndlow:', np.round(tbias_bndlow,2), 'mb_mwea:', np.round(mb_mwea,2))

                    # Upper bound
                    while mb_mwea > mb_obs_mwea and modelprms['tbias'] < 20:
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        if debug:
                            print('  tc:', np.round(modelprms['tbias'],2), 'mb_mwea:', np.round(mb_mwea,2))
                        tbias_bndhigh = modelprms['tbias']

                    # Single parameter optimizer (computationally more efficient and less prone to fail)
                    modelprms_subset = {'kp':kp_opt, 
                                        'ddfsnow': ddfsnow_opt, 
                                        'tbias': modelprms['tbias'] - tbias_step/2}
                    tbias_bnds = (tbias_bndhigh-tbias_step, tbias_bndhigh)
                    modelprms_opt, mb_mwea = single_param_optimizer(
                            modelprms_subset, mb_obs_mwea, prm2opt='tbias', tbias_bnds=tbias_bnds, debug=debug)
                    
                    # Update parameter values
                    tbias_opt = modelprms_opt['tbias']
                    modelprms['tbias'] = tbias_opt
                    if debug:
                        print('  tbias:', np.round(tbias_opt,3), 'mb_mwea:', np.round(mb_mwea,3))

                else:
                    tbias_opt = modelprms['tbias']
                
                    
                # Export model parameters
                modelprms = modelprms_opt
                for vn in ['ddfice', 'ddfsnow', 'kp', 'precgrad', 'tbias', 'tsnow_threshold']:
                    modelprms[vn] = [modelprms[vn]]
                modelprms['mb_mwea'] = [mb_mwea]
                modelprms['mb_obs_mwea'] = [mb_obs_mwea]
                modelprms['mb_obs_mwea_err'] = [mb_obs_mwea_err]

                modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                modelprms_fp = (pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                + '/')
                if not os.path.exists(modelprms_fp):
                    os.makedirs(modelprms_fp, exist_ok=True)
                modelprms_fullfn = modelprms_fp + modelprms_fn
                if os.path.exists(modelprms_fullfn):
                    with open(modelprms_fullfn, 'rb') as f:
                        modelprms_dict = pickle.load(f)
                    modelprms_dict[pygem_prms.option_calibration] = modelprms
                else:
                    modelprms_dict = {pygem_prms.option_calibration: modelprms}
                with open(modelprms_fullfn, 'wb') as f:
                    pickle.dump(modelprms_dict, f)
                    
                    
            #%% ===== MODIFIED HUSS AND HOCK (2015) CALIBRATION =====
            # used in Rounce et al. (2020; MCMC paper)
            # - precipitation factor, then temperature bias (no ddfsnow)
            # - ranges different
            elif pygem_prms.option_calibration == 'HH2015mod':
                tbias_init = pygem_prms.tbias_init
                tbias_step = pygem_prms.tbias_step
                kp_init = pygem_prms.kp_init
                kp_bndlow = pygem_prms.kp_bndlow
                kp_bndhigh = pygem_prms.kp_bndhigh
                ddfsnow_init = pygem_prms.ddfsnow_init
                
                # ----- Initialize model parameters -----
                modelprms['tbias'] = tbias_init
                modelprms['kp'] = kp_init
                modelprms['ddfsnow'] = ddfsnow_init
                modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                
                # ----- FUNCTIONS -----
                def objective(modelprms_subset):
                    """ Objective function for mass balance data (mimize difference between model and observation).

                    Parameters
                    ----------
                    modelprms_subset : list of model parameters [kp, ddfsnow, tbias]
                    """
                    # Subset of model parameters used to reduce number of constraints required
                    modelprms['kp'] = modelprms_subset[0]
                    modelprms['tbias'] = tbias_init
                    if len(modelprms_subset) > 1:
                        modelprms['tbias'] = modelprms_subset[1]  
                    # Mass balance
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    # Difference with observation (mwea)
                    mb_dif_mwea_abs = abs(mb_obs_mwea - mb_mwea)
                    return mb_dif_mwea_abs


                def run_objective(modelprms_init, mb_obs_mwea, modelprms_bnds=None, 
                                  run_opt=True, eps_opt=pygem_prms.eps_opt,
                                  ftol_opt=pygem_prms.ftol_opt):
                    """ Run the optimization for the single glacier objective function.

                    Parameters
                    ----------
                    modelparams_init : list of model parameters to calibrate [kp, ddfsnow, tbias]
                    kp_bnds, tbias_bnds, ddfsnow_bnds, precgrad_bnds : tuples (lower & upper bounds)
                    run_opt : Boolean statement run optimization or bypass optimization and run with initial parameters

                    Returns
                    -------
                    modelparams : model parameters dict and specific mass balance (mwea)
                    """
                    # Run the optimization
                    if run_opt:
                        modelprms_opt = minimize(objective, modelprms_init, method=pygem_prms.method_opt,
                                                bounds=modelprms_bnds, options={'ftol':ftol_opt, 'eps':eps_opt})
                        # Record the optimized parameters
                        modelprms_subset = modelprms_opt.x
                    else:
                        modelprms_subset = modelprms.copy()
                    modelprms['kp'] = modelprms_subset[0]
                    if len(modelprms_subset) == 2:
                        modelprms['tbias'] = modelprms_subset[1]                    
                    # Re-run the optimized parameters in order to see the mass balance
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    return modelprms, mb_mwea


                # ----- Temperature bias bounds -----
                tbias_bndhigh = 0
                # Tbias lower bound based on no positive temperatures
                tbias_bndlow = (-1 * (gdir.historical_climate['temp'] + gdir.historical_climate['lr'] *
                                (fls[0].surface_h.min() - gdir.historical_climate['elev'])).max())
                modelprms['tbias'] = tbias_bndlow
                mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                if debug:
                    print('  tbias_bndlow:', np.round(tbias_bndlow,2), 'mb_mwea:', np.round(mb_mwea,2))
                # Tbias upper bound (based on kp_bndhigh)
                modelprms['kp'] = kp_bndhigh
                
                while mb_mwea > mb_obs_mwea and modelprms['tbias'] < 20:
                    modelprms['tbias'] = modelprms['tbias'] + 1
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    if debug:
                        print('  tc:', np.round(modelprms['tbias'],2), 'mb_mwea:', np.round(mb_mwea,2))
                    tbias_bndhigh = modelprms['tbias']

                # ===== ROUND 1: PRECIPITATION FACTOR =====
                # Adjust bounds based on range of temperature bias
                if tbias_init > tbias_bndhigh:
                    tbias_init = tbias_bndhigh
                elif tbias_init < tbias_bndlow:
                    tbias_init = tbias_bndlow
                modelprms['tbias'] = tbias_init
                modelprms['kp'] = kp_init

                tbias_bndlow_opt = tbias_init
                tbias_bndhigh_opt = tbias_init

                # Constrain bounds of precipitation factor and temperature bias
                mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                nbinyears_negmbclim = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                  return_tbias_mustmelt=True)

                if debug:
                    print('\ntbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                          'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))

                # Adjust lower or upper bound based on the observed mass balance
                test_count = 0
                if mb_mwea > mb_obs_mwea:
                    if debug:
                        print('increase tbias, decrease kp')
                    kp_bndhigh = 1
                    # Check if lower bound causes good agreement
                    modelprms['kp'] = kp_bndlow
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    while mb_mwea > mb_obs_mwea and test_count < 20:
                        # Update temperature bias
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        # Update bounds
                        tbias_bndhigh_opt = modelprms['tbias']
                        tbias_bndlow_opt = modelprms['tbias'] - tbias_step
                        # Compute mass balance
                        mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        if debug:
                            print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))
                        test_count += 1
                else:
                    if debug:
                        print('decrease tbias, increase kp')
                    kp_bndlow = 1
                    # Check if upper bound causes good agreement
                    modelprms['kp'] = kp_bndhigh
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    
                    while mb_obs_mwea > mb_mwea and test_count < 20:
                        # Update temperature bias
                        modelprms['tbias'] = modelprms['tbias'] - tbias_step
                        # If temperature bias is at lower limit, then increase precipitation factor
                        if modelprms['tbias'] <= tbias_bndlow:
                            modelprms['tbias'] = tbias_bndlow
                            if test_count > 0:
                                kp_bndhigh = kp_bndhigh + 1
                                modelprms['kp'] = kp_bndhigh
                        # Update bounds (must do after potential correction for lower bound)
                        tbias_bndlow_opt = modelprms['tbias']
                        tbias_bndhigh_opt = modelprms['tbias'] + tbias_step
                        # Compute mass balance
                        mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        if debug:
                            print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'mb_mwea:', np.round(mb_mwea,2), 'obs_mwea:', np.round(mb_obs_mwea,2))
                        test_count += 1

                # ----- RUN OPTIMIZATION WITH CONSTRAINED BOUNDS -----
                kp_bnds = (kp_bndlow, kp_bndhigh)
                kp_init = kp_init
                
                tbias_bnds = (tbias_bndlow_opt, tbias_bndhigh_opt)
                tbias_init = np.mean([tbias_bndlow_opt, tbias_bndhigh_opt])

                if debug:
                    print('tbias bounds:', tbias_bnds)
                    print('kp bounds:', kp_bnds)
                
                # Set up optimization for only the precipitation factor
                if tbias_bndlow_opt == tbias_bndhigh_opt:
                    modelprms_subset = [kp_init]
                    modelprms_bnds = (kp_bnds,)
                # Set up optimization for precipitation factor and temperature bias
                else:
                    modelprms_subset = [kp_init, tbias_init]
                    modelprms_bnds = (kp_bnds, tbias_bnds)
                    
                # Run optimization
                modelparams_opt, mb_mwea = run_objective(modelprms_subset, mb_obs_mwea, 
                                                        modelprms_bnds=modelprms_bnds, ftol_opt=1e-3)

                kp_opt = modelparams_opt['kp']
                tbias_opt = modelparams_opt['tbias']
                if debug:
                    print('mb_mwea:', np.round(mb_mwea,2), 'obs_mb:', np.round(mb_obs_mwea,2),
                          'kp:', np.round(kp_opt,2), 'tbias:', np.round(tbias_opt,2), '\n\n')

                # Export model parameters
                modelprms = modelparams_opt
                for vn in ['ddfice', 'ddfsnow', 'kp', 'precgrad', 'tbias', 'tsnow_threshold']:
                    modelprms[vn] = [modelprms[vn]]
                modelprms['mb_mwea'] = [mb_mwea]
                modelprms['mb_obs_mwea'] = [mb_obs_mwea]
                modelprms['mb_obs_mwea_err'] = [mb_obs_mwea_err]

                modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                modelprms_fp = (pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                + '/')
                if not os.path.exists(modelprms_fp):
                    os.makedirs(modelprms_fp, exist_ok=True)
                modelprms_fullfn = modelprms_fp + modelprms_fn
                if os.path.exists(modelprms_fullfn):
                    with open(modelprms_fullfn, 'rb') as f:
                        modelprms_dict = pickle.load(f)
                    modelprms_dict[pygem_prms.option_calibration] = modelprms
                else:
                    modelprms_dict = {pygem_prms.option_calibration: modelprms}
                with open(modelprms_fullfn, 'wb') as f:
                    pickle.dump(modelprms_dict, f)

        else:
            # LOG FAILURE
            fail_fp = pygem_prms.output_filepath + 'cal_fail/' + glacier_str.split('.')[0].zfill(2) + '/'
            if not os.path.exists(fail_fp):
                os.makedirs(fail_fp, exist_ok=True)
            txt_fn_fail = glacier_str + "-cal_fail.txt"
            with open(fail_fp + txt_fn_fail, "w") as text_file:
                if pygem_prms.option_calibration=='MCMC' and pygem_prms.option_calib_binned_dh:
                    text_file.write(glacier_str + ' had no compatible surface elevation data.')
                else:
                    text_file.write(glacier_str + ' had no flowlines or mb_data.')                    

    # Global variables for Spyder development
    if args.num_simultaneous_processes == 1:
        global main_vars
        main_vars = inspect.currentframe().f_locals


#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

#    cfg.initialize()
#    cfg.PARAMS['use_multiprocessing']  = False
#    if not 'pygem_modelprms' in cfg.BASENAMES:
#        cfg.BASENAMES['pygem_modelprms'] = ('pygem_modelprms.pkl', 'PyGEM model parameters')

    # RGI glacier number
    if args.rgi_glac_number:
        glac_no = args.rgi_glac_number
    elif args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            glac_no = pickle.load(f)
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
                include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater, 
                min_glac_area_km2=pygem_prms.min_glac_area_km2)
        glac_no = list(main_glac_rgi_all['rgino_str'].values)

    # Number of cores for parallel processing
    if args.num_simultaneous_processes > 1:
        num_cores = int(np.min([len(glac_no), args.num_simultaneous_processes]))
    else:
        num_cores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = modelsetup.split_list(glac_no, n=num_cores)

    # Read GCM names from argument parser
    gcm_name = args.ref_gcm_name
    print('Processing:', gcm_name)
    
    # Pack variables for multiprocessing
    list_packed_vars = []
    for count, glac_no_lst in enumerate(glac_no_lsts):
        list_packed_vars.append([count, glac_no_lst, gcm_name])

    # Parallel processing
    if num_cores > 1:
        print('Processing in parallel with ' + str(num_cores) + ' cores...')
        with multiprocessing.Pool(num_cores) as p:
            p.map(main,list_packed_vars)
    # If not in parallel, then only should be one loop
    else:
        # Loop through the chunks and export bias adjustments
        for n in range(len(list_packed_vars)):
            main(list_packed_vars[n])



    print('Total processing time:', time.time()-time_start, 's')


# #%% ===== PLOTTING AND PROCESSING FOR MODEL DEVELOPMENT =====
#     # Place local variables in variable explorer
#     if args.option_parallels == 0:
#         main_vars_list = list(main_vars.keys())
#         gcm_name = main_vars['gcm_name']
#         main_glac_rgi = main_vars['main_glac_rgi']
#         if pygem_prms.hyps_data in ['Huss', 'Farinotti']:
#             main_glac_hyps = main_vars['main_glac_hyps']
#             main_glac_icethickness = main_vars['main_glac_icethickness']
#             main_glac_width = main_vars['main_glac_width']
#         dates_table = main_vars['dates_table']
#         gcm_temp = main_vars['gcm_temp']
#         gcm_tempstd = main_vars['gcm_tempstd']
#         gcm_prec = main_vars['gcm_prec']
#         gcm_elev = main_vars['gcm_elev']
#         gcm_lr = main_vars['gcm_lr']
#         gcm_temp_lrglac = main_vars['gcm_lr']
#         glacier_rgi_table = main_vars['glacier_rgi_table']
#         glacier_str = main_vars['glacier_str']
#         if pygem_prms.hyps_data in ['OGGM']:
#             gdir = main_vars['gdir']
#             fls = main_vars['fls']
#             elev_bins = fls[0].surface_h
#             width_initial = fls[0].widths_m / 1000
#             glacier_area_initial = width_initial * fls[0].dx / 1000
#             if pygem_prms.use_calibrated_modelparams:
#                 modelprms_dict = main_vars['modelprms_dict']

# #%%
# import numpy as np
# import pickle
# # fullfn = '/Users/drounce/Documents/HiMAT/R11_rgi_glac_number_1-1000glac_batch_0.pkl'
# # with open(fullfn, 'rb') as f:
# #     A = pickle.load(f)
# modelprms_fullfn = '/Users/drounce/Documents/HiMAT/Output/calibration/01/1.10689-modelprms_dict.pkl'
# with open(modelprms_fullfn, 'rb') as f:
#     modelprms_dict = pickle.load(f)
# print('mcmc:', modelprms_dict['MCMC']['mb_obs_mwea'], np.mean(modelprms_dict['MCMC']['mb_mwea']['chain_0']))
# print('emulator:', modelprms_dict['emulator'])
# #%%
# print(modelprms_dict[pygem_prms.option_calibration][‘kp’])
# print('obs:', modelprms_dict['MCMC']['mb_obs_mwea'])
# print('\nEmulator:')
# print(np.mean(modelprms_dict['MCMC']['mb_mwea']['chain_0']))
# print(np.round(np.median(modelprms_dict['MCMC']['mb_mwea']['chain_0']),3), 
#       np.round(median_abs_deviation(modelprms_dict['MCMC']['mb_mwea']['chain_0']),3))
# print(np.round(np.median(modelprms_dict['MCMC']['tbias']['chain_0']),3), 
#       np.round(median_abs_deviation(modelprms_dict['MCMC']['tbias']['chain_0']),3))
# print(np.round(np.median(modelprms_dict['MCMC']['kp']['chain_0']),3), 
#       np.round(median_abs_deviation(modelprms_dict['MCMC']['kp']['chain_0']),3))
# print(np.round(np.median(modelprms_dict['MCMC']['ddfsnow']['chain_0']),6), 
#       np.round(median_abs_deviation(modelprms_dict['MCMC']['ddfsnow']['chain_0']),6))
# print('\nFull sim:')
# print(np.round(np.median(modelprms_dict['MCMC_fullsim']['mb_mwea']['chain_0']),3), 
#       np.round(median_abs_deviation(modelprms_dict['MCMC_fullsim']['mb_mwea']['chain_0']),3))
# print(np.round(np.median(modelprms_dict['MCMC_fullsim']['tbias']['chain_0']),3), 
#       np.round(median_abs_deviation(modelprms_dict['MCMC_fullsim']['tbias']['chain_0']),3))
# print(np.round(np.median(modelprms_dict['MCMC_fullsim']['kp']['chain_0']),3), 
#       np.round(median_abs_deviation(modelprms_dict['MCMC_fullsim']['kp']['chain_0']),3))
# print(np.round(np.median(modelprms_dict['MCMC_fullsim']['ddfsnow']['chain_0']),6), 
#       np.round(median_abs_deviation(modelprms_dict['MCMC_fullsim']['ddfsnow']['chain_0']),6))