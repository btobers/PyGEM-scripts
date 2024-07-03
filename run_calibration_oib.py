"""Run model calibration using Operation IceBridge data

psuedocode:
1. ingest OIB data
2. get mask of OGGM bins for which we have OIB data at certain time steps (different surveys will have different masks)
3. create emulator for total mass balance over unmasked bins
4. compare emulator with OIB observations to make sure there's agreement
5. store model parameters for forward runs
"""


# Built-in libraries
import argparse
#import collections
import inspect
import warnings
import importlib
import multiprocessing
import os
import glob
import sys
import time
import cftime
import random
import json
import datetime
from functools import partial
# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, fmin_bfgs
from scipy import stats, signal
from scipy.interpolate import interp1d
import cmocean
#import xarray as xr

# Local libraries
import pygem
import pygem_input as pygem_prms
from pygem import class_climate
from pygem.massbalance import PyGEMMassBalance
#from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory, single_flowline_glacier_directory_with_calving
import pygem.pygem_modelsetup as modelsetup
from pygem.shop import debris, mbdata, icethickness

import oggm
oggm_version = float(oggm.__version__[0:3])
from oggm import cfg
from oggm import graphics
from oggm import tasks
from oggm import utils
from oggm import workflow
#from oggm.core import climate
from oggm.core.flowline import FluxBasedModel
#from oggm.core.inversion import calving_flux_from_depth
if oggm_version > 1.301:
    from oggm.core.massbalance import apparent_mb_from_any_mb # Newer Version of OGGM
else:
    from oggm.core.climate import apparent_mb_from_any_mb # Older Version of OGGM

# Model-specific libraries
if pygem_prms.option_calibration in ['emulator', 'MCMC']:
    import torch
    import gpytorch
    import sklearn.model_selection
if pygem_prms.option_calibration in ['MCMC']:
    import pymc
    from pymc import deterministic

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
    option_parallels (optional) : int
        switch to use parallels or not
    rgi_glac_number_fn : str
        filename of .pkl file containing a list of glacier numbers which is used to run batches on the supercomputer
    rgi_glac_number : str
        rgi glacier number to run for supercomputer
    option_ordered : int
        option to keep glaciers ordered or to grab every n value for the batch
        (the latter helps make sure run times on each core are similar as it removes any timing differences caused by
         regional variations)
    progress_bar : int
        Switch for turning the progress bar on or off (default = 0 (off))
    debug : int
        Switch for turning debug printing on or off (default = 0 (off))

    Returns
    -------
    Object containing arguments and their respective values.
    """
    parser = argparse.ArgumentParser(description="run calibration in parallel")
    # add arguments
    parser.add_argument('-ref_gcm_name', action='store', type=str, default=pygem_prms.ref_gcm_name,
                        help='reference gcm name')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store_true',
                        help='Switch to use or not use parallels (1 - use parallels, 0 - do not)')
    parser.add_argument('-rgi_glac_number_fn', action='store', type=str, default=None,
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-option_ordered', action='store', type=int, default=1,
                        help='switch to keep lists ordered or not')
    parser.add_argument('-progress_bar', action='store', type=int, default=0,
                        help='Boolean for the progress bar to turn it on or off (default 0 is off)')
    parser.add_argument('-debug', action='store_true',
                        help='Flag for debugging (default is off)')
    parser.add_argument('-rgi_glac_number', action='store', type=str, default=None,
                        help='rgi glacier number for supercomputer')
    parser.add_argument('-option_calibration', action='store', type=str, default=pygem_prms.option_calibration,
                        help='calibration option')
    parser.add_argument('-kp', action='store', type=float, default=None,
                        help='Precipitation bias')
    parser.add_argument('-tbias', action='store', type=float, default=None,
                        help='Temperature bias')
    parser.add_argument('-ddfsnow', action='store', type=float, default=None,
                        help='Degree-day factor of snow')
    parser.add_argument('-errortype', action='store', type=str, default='rmse',
                        help='Error metric (rmse, mad, l1, l2)')
    parser.add_argument('-weighted', action='store_true',
                        help='Flag to weight misfit minimization by area')
    return parser


def dict_append(dictionary, keys=None, vals=None):
    """
    Simple function to append values to a dictionary - useful for updating emulator parameters
    """
    if isinstance(keys,list) and isinstance(vals,list) and (len(keys)==len(vals)):
        for _i,k in enumerate(keys):
            if k not in dictionary.keys():
                dictionary[k] = []
            try:
                if isinstance(vals[_i],list):
                    dictionary[k] += vals[_i]
                else:
                    dictionary[k].append(vals[_i])
            except Exception as err:
                print(f'dict_append error: {err}')

    return dictionary


def index_closest(arr,val):
    """
    get index in an array or list closest to a specified value
    """
    return np.abs(np.asarray(arr) - val).argmin()
    

def mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=None, t1=None, t2=None):
    """
    Run the mass balance and calculate the mass balance [mwea]

    Parameters
    ----------
    option_areaconstant : Boolean

    Returns
    -------
    nbinyears_negmbclim: int
        count, number of bins where there is a negative annual climatic mass balance across all model years
    mb_mwea : float
        mass balance [m w.e. a-1]
    """
    # RUN MASS BALANCE MODEL
    mbmod = PyGEMMassBalance(gdir, modelprms, glacier_rgi_table, fls=fls,
                             debug=pygem_prms.debug_mb, debug_refreeze=pygem_prms.debug_refreeze)
    years = np.arange(0, int(gdir.dates_table.shape[0]/12))
    for year in years:
        mbmod.get_annual_mb(fls[0].surface_h, fls=fls, fl_id=0, year=year, debug=False)

    # Number of years and bins with negative climatic mass balance
    nbinyears_negmbclim =  len(np.where(mbmod.glac_bin_massbalclim_annual < 0)[0])
    # specific mass balance
    t1_idx = gdir.mbdata['t1_idx']
    t2_idx = gdir.mbdata['t2_idx']
    nyears = gdir.mbdata['nyears']
    mb_mwea = mbmod.glac_wide_massbaltotal[t1_idx:t2_idx+1].sum() / mbmod.glac_wide_area_annual[0] / nyears

    return nbinyears_negmbclim, mb_mwea


def binned_mb_calc(gdir, modelprms, glacier_rgi_table, fls=None, glen_a_multiplier=None, fs=None, debug=False):
    """
    Run the ice thickness inversion and mass balance model to get binned annual ice thickness evolution
    """
    if debug:
        print(f'binned_mb_calc() modelprms: {modelprms}')
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
                                    water_level=water_level
                                    )

        # run glacier dynamics model forward
        ev_model.run_until_and_store(nyears)

        t1_idx = gdir.mbdata['t1_idx']
        t2_idx = gdir.mbdata['t2_idx']
        nyears = gdir.mbdata['nyears']
        mb_mwea = mbmod.glac_wide_massbaltotal[t1_idx:t2_idx+1].sum() / mbmod.glac_wide_area_annual[0] / nyears


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

        if debug:
            print(f'specific mass balance = {np.round(mb_mwea,2)} mwea')
        return nfls[0].surface_h, mbmod.glac_bin_massbalclim, mbmod.glac_bin_icethickness_annual, mb_mwea


def get_bin_thick_monthly(bin_massbalclim_monthly, bin_thick_annual):
    """
    funciton to calculate the monthly binned ice thickness
    from annual climatic mass balance and annual ice thickness

    monthly binned ice thickness is determined assuming 
    the flux divergence is constant throughout the year.

    Inputs
    ----------
    bin_massbalclim_monthly : float
        ndarray containing the climatic mass balance for each model month computed by PyGEM
        shape : [#elevbins, #months]
    bin_thick_annual : float
        ndarray containing the average (or median) binned ice thickness at computed by PyGEM
        shape : [#elevbins, #years]

    Outputs
    -------
    bin_thick_monthly: float
        ndarray containing the binned monthly ice thickness
        shape : [#elevbins, #months]
    """
    # get annual climatic mass balance from monthly climatic mass balance - requires reshaping monthly binned values and summing every 12 months
    bin_massbalclim_annual = bin_massbalclim_monthly.reshape(bin_massbalclim_monthly.shape[0],bin_massbalclim_monthly.shape[1]//12,-1).sum(2)

    # bin_thick_annual = bin_thick_annual[:,:-1]
    # get change in thickness from previous year for each elevation bin
    delta_thick_annual = np.diff(bin_thick_annual, axis=-1)

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
                (bin_massbalclim_monthly * 
                (pygem_prms.density_ice / 
                pygem_prms.density_water)) - 
                flux_div_monthly)

    # get binned monthly thickness = running thickness change + initial thickness
    running_delta_thick_monthly = np.cumsum(delta_thick_monthly, axis=-1)
    bin_thick_monthly =  running_delta_thick_monthly + bin_thick_annual[:,0][:,np.newaxis] 

    return bin_thick_monthly


if pygem_prms.option_calibration in ['MCMC', 'emulator']:
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


    def create_emulator(glacier_str, sims_dict, y_cn, 
                        X_cns=['tbias','kp','ddfsnow','nbinyrs_negmbclim','time','bin_h_init'], 
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
        em_t0 = time.time()
        # This is required for the supercomputer such that resources aren't stolen from other cpus
        torch.set_num_threads(1)
        
        assert y_cn in sims_dict.keys(), 'emulator error: y_cn not in sims_dict'
        em_mod_fp = em_fp + 'models/' + glacier_str.split('.')[0].zfill(2) + '/'

        ##################        
        ### get X data ###
        ##################
        X_cns = [cn for cn in X_cns if cn in sims_dict]
        Xs = [sims_dict[cn] for cn in X_cns]

        # convert to numpy arrays
        X = np.column_stack((Xs))
        y = np.array(sims_dict[y_cn])

        # remove any nan's
        nanmask  = ~np.isnan(y)
        X = X[nanmask,:]
        y = y[nanmask]

        if debug:
            print(f'Calibration x-parameters: {", ".join(X_cns)}')
            print(f'Calibration y-parameters: {y_cn}')
            print(f'X:\n{X}')
            print(f'X-shape:\n{X.shape}\n')
            print(f'y:\n{y}')
            print(f'y-shape:\n{y.shape}')

        ##################
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
                ax.axline((0, 0), slope=1, c='k')
                ax.plot(y_test.numpy()[idx], y_pred.mean.numpy()[idx], 'k*')
                ax.fill_between(y_test.numpy()[idx], lower.numpy()[idx], upper.numpy()[idx], alpha=0.5)
                ax.set_xlabel('y_test')
                ax.set_ylabel('y_pred')
                ax.set_ylim([-3,3])
                f.tight_layout()
                f.savefig(f'{em_mod_fp}/{glacier_str}_emulator_prior_thick.png')
                plt.close()

    
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
            if debug and i%100==0:
                print(f'iteration = {i}, loss = {loss.item()}, covariance lengthscale = {model.covar_module.base_kernel.lengthscale[0].item()}, noise = {model.likelihood.noise.item()}')
            optimizer.step()
    
        # Plot posterior distributions (with test data on x-axis)
        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()
        with torch.no_grad():#, gpytorch.settings.fast_pred_var():
            y_pred = likelihood(model(X_test))
    
        idx = np.argsort(y_test.numpy())

        print('emulator runtime:', time.time()-em_t0, 's')
    
        with torch.no_grad():
            lower, upper = y_pred.confidence_region()
    
            if debug:
                f, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.plot(y_test.numpy()[idx], y_pred.mean.numpy()[idx], 'k*')
                ax.fill_between(y_test.numpy()[idx], lower.numpy()[idx], upper.numpy()[idx], 
                                alpha=0.5)
                ax.axline((0, 0), slope=1, c='k')
                ax.set_xlabel('y_test')
                ax.set_ylabel('y_pred')
                ax.set_ylim([-3,3])
                f.tight_layout()
                # plt.show()
                f.savefig(f'{em_mod_fp}/{glacier_str}_emulator_posterior_thick.png')
                plt.close()

    
        # if debug:
        for batman in [0]:
            # Compare user-defined parameter sets within the emulator
            tbias_set = (np.arange(-7,4,0.5)).reshape(-1,1)
            kp_set = np.zeros(tbias_set.shape) + 1
            ddf_set = np.zeros(tbias_set.shape) + 0.0041
            sets = []

            # Convert the array into a 1D array of complex numbers
            # where the real part represents the values from the first column
            # and the imaginary part represents the values from the second column
            complex_array = X[:, -2] + 1j * X[:, -1]

            # Find the unique pairs of values
            unique_pairs = np.unique(complex_array)

            # Separate the real and imaginary parts to get the unique pairs
            unique_pairs_real = unique_pairs.real
            unique_pairs_imag = unique_pairs.imag

            if y_cn=='bin_thick_monthly':
                for t, h in zip(unique_pairs_real, unique_pairs_imag):
                    # test set for time and elevation bin
                    time_set = np.zeros(tbias_set.shape) + t
                    elev_set = np.zeros(tbias_set.shape) + h
                    sets.append(np.hstack((tbias_set, kp_set, ddf_set, time_set, elev_set)))
            else:
                sets.append(np.hstack((tbias_set, kp_set, ddf_set)))

            # for i,s in enumerate(sets):
            #     # normalize test set values
            #     modelprms_set_norm = (s - X_mean) / X_std
        
            #     y_set_norm = model(torch.tensor(modelprms_set_norm).to(torch.float)).mean.detach().numpy()
            #     y_set = y_set_norm * y_std + y_mean
        
            #     f, ax = plt.subplots(1, 1, figsize=(4, 4))
            #     # kp_1_idx = np.where(sim_dict['kp'] == 1)[0]
            #     kp_1_idx = np.where(X[:,1] == 1)[0]
            #     # ax.plot(sims_df.loc[kp_1_idx,'tbias'], sims_df.loc[kp_1_idx,y_cn])
            #     ax.plot(X[kp_1_idx,0], y[kp_1_idx])
            #     ax.plot(tbias_set,y_set,'.')
            #     ax.set_xlabel('tbias (degC)')
            #     if y_cn == 'mb_mwea':
            #         ax.set_ylabel('PyGEM MB (mwea)')
            #     elif y_cn == 'nbinyrs_negmbclim':
            #         ax.set_ylabel('nbinyrs_negmbclim (-)')
            #     elif y_cn == 'bin_thick_monthly':
            #         ax.set_ylabel('Ice Thickness (m)')
            #         t = s[0,-2]
            #         h = s[0,-1]
            #         ax.plot([],[],label=f'{np.round(h).astype(int)} m, month {int(t)}')
            #         ax.legend(handlelength=0, loc='upper right', borderaxespad=0, fancybox=False)

            #     f.tight_layout()
            #     f.savefig(f'{em_mod_fp}/{glacier_str}_emulator_{int(t)}_{np.round(h,2)}m.png')
            #     plt.close()
                # plt.show()
    
            # Compare the modeled and emulated mass balances for entire test set
            y_em_norm = model(torch.tensor(X_norm).to(torch.float)).mean.detach().numpy()
            y_em = y_em_norm * y_std + y_mean
    
            f, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.plot(y, y_em, '.')
            ax.plot([y.min(),y.max()], [y.min(), y.max()],c='k',lw=1)
            if y_cn == 'mb_mwea':
                ax.set_ylabel('emulator MB (mwea)')
                ax.set_xlabel('PyGEM MB (mwea)')
                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
            elif y_cn == 'nbinyrs_negmbclim':
                ax.set_ylabel('emulator nbinyrs_negmbclim (-)')
                ax.set_xlabel('PyGEM nbinyrs_negmbclim (-)')
            elif y_cn == 'bin_thick_monthly':
                ax.set_ylabel('emulator ice thickness (m)')
                ax.set_xlabel('PyGEM ice thickness (m)')
                f.tight_layout()
                f.savefig(f'{em_mod_fp}/{glacier_str}_emulator_v_model_thick.png')
                # plt.show()
                plt.close()

        # ----- EXPORT EMULATOR -----
        # Save emulator (model state, x_train, y_train, etc.)
        em_mod_fn = glacier_str + '-emulator-' + y_cn + '.pth'
        em_mod_fp = em_fp + 'models/' + glacier_str.split('.')[0].zfill(2) + '/'
        if not os.path.exists(em_mod_fp):
            os.makedirs(em_mod_fp, exist_ok=True)
        torch.save(model.state_dict(), em_mod_fp + em_mod_fn)
        # Extra required datasets
        em_extra_dict = {'X_train': X_train,
                         'X_mean': X_mean,
                         'X_std': X_std,
                         'y_train': y_train,
                         'y_mean': y_mean,
                         'y_std': y_std}
        em_extra_fn = em_mod_fn.replace('.pth','_extra.pkl')
        with open(em_mod_fp + em_extra_fn, 'wb') as f:
            pickle.dump(em_extra_dict, f)
            
        return X_train, X_mean, X_std, y_train, y_mean, y_std, likelihood, model
        

def get_rgi7id(rgi6id='', debug=False):
    """
    return RGI version 7 glacier id for a given RGI version 6 id
    """
    rgi6id = rgi6id.split('.')[0].zfill(2) + '.' + rgi6id.split('.')[1]
    # get appropriate RGI7 Id from PyGEM RGI6 Id
    rgi7_6_df = pd.read_csv(pygem_prms.rgi_fp + '/RGI2000-v7.0-G-01_alaska-rgi6_links.csv')
    rgi7_6_df['rgi7_id'] = rgi7_6_df['rgi7_id'].str.split('RGI2000-v7.0-G-').str[1]
    rgi7_6_df['rgi6_id'] = rgi7_6_df['rgi6_id'].str.split('RGI60-').str[1]
    rgi7id = rgi7_6_df.loc[lambda rgi7_6_df: rgi7_6_df['rgi6_id'] == rgi6id,'rgi7_id'].tolist()[0]
    if debug:
        print(f'RGI6:{rgi6id} -> RGI7:{rgi7id}')
    return rgi7id


def date_check(dt_obj):
    """
    if survey date in given month <daysinmonth/2 assign it to beginning of month, else assign to beginning of next month (for consistency with monthly PyGEM timesteps)
    """
    dim = pd.Series(dt_obj).dt.daysinmonth.iloc[0]
    if dt_obj.day < dim // 2:
        dt_obj_ = datetime.datetime(year=dt_obj.year, month=dt_obj.month, day=1)
    else:
        dt_obj_ = datetime.datetime(year=dt_obj.year, month=dt_obj.month+1, day=1)
    return dt_obj_


def load_oib(rgi7id):
    """
    load Operation IceBridge data
    """
    oib_fpath = glob.glob(pygem_prms.oib_fp  + f'/diffstats5_*{rgi7id}*.json')
    if len(oib_fpath)==0:
        return
    else:
        oib_fpath = oib_fpath[0]
    # load diffstats file
    with open(oib_fpath, 'rb') as f:
        oib_dict = json.load(f)
    return oib_dict


def oib_filter_on_pixel_count(arr, pctl = 15):
    """
    filter oib diffs by perntile pixel count
    """
    arr=arr.astype(float)
    arr[arr==0] = np.nan
    mask = arr < np.nanpercentile(arr,pctl)
    arr[mask] = np.nan
    return arr


def oib_terminus_mask(survey_date, cop30_diffs, debug=False):
    """
    create mask of missing terminus ice using last oib survey
    """
    try:
        # find peak we'll bake in the assumption that terminus thickness has decreased over time - we'll thus look for a trough if yr>=2013 (cop30 date)
        if survey_date.year<2013:
            arr = cop30_diffs
        else:
            arr = -1*cop30_diffs
        pk = signal.find_peaks(arr, distance=200)[0][0]
        if debug:
            plt.figure()
            plt.plot(cop30_diffs)
            plt.axvline(pk,c='r')
            plt.show()

        return(np.arange(0,pk+1,1))

    except Exception as err:
        if debug:
            print(f'_filter_terminus_missing_ice error: {err}')
    return []


def get_oib_diffs(oib_dict, debug=False):
    """
    loop through OIB dataset, get double differences
    diffs_stacked: np.ndarray (#bins, #surveys)
    """
    seasons = list(set(oib_dict.keys()).intersection(['march','may','august']))
    cop30_diffs_list = [] # instantiate list to hold median binned survey differences from cop30
    oib_dates = [] # instantiate list to hold survey dates
    for ssn in seasons:
        for yr in list(oib_dict[ssn].keys()):
            # get survey date
            doy_int = int(np.ceil(oib_dict[ssn][yr]['mean_doy']))
            dt_obj = datetime.datetime.strptime(f'{int(yr)}-{doy_int}', '%Y-%j')
            oib_dates.append(date_check(dt_obj))
            # get survey data and filter by pixel count
            diffs = np.asarray(oib_dict[ssn][yr]['bin_vals']['bin_median_diffs_vec'])
            diffs = oib_filter_on_pixel_count(diffs, 15)
            cop30_diffs_list.append(diffs)
    # sort by survey dates
    inds = np.argsort(oib_dates).tolist()
    oib_dates = [oib_dates[i] for i in inds]
    cop30_diffs_list = [cop30_diffs_list[i] for i in inds]
    # filter missing ice at terminus based on last survey
    terminus_mask = oib_terminus_mask(oib_dates[-1], cop30_diffs_list[-1], debug=False)    
    if debug:
        print(f'OIB survey dates:\n{", ".join([str(dt.year)+"-"+str(dt.month)+"-"+str(dt.day) for dt in oib_dates])}')
    # do double differencing
    diffs_stacked = np.column_stack(cop30_diffs_list)
    # apply terminus mask across all surveys
    diffs_stacked[terminus_mask,:] = np.nan
    # get bin centers
    bin_centers = (np.asarray(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_start_vec']) + 
                np.asarray(oib_dict[ssn][list(oib_dict[ssn].keys())[0]]['bin_vals']['bin_stop_vec'])) / 2
    bin_area = oib_dict['aad_dict']['hist_bin_areas_m2']

    return bin_centers, bin_area, diffs_stacked, pd.Series(oib_dates)


def oib_model_fx(params, gdir, modelprms, glacier_rgi_table, fls, glen_a_mult, fs):
    modelprms['kp'] = params[0]
    modelprms['tbias'] = params[1]
    modelprms['ddfsnow'] = params[2]
    surf_h_init, bin_massbalclim_monthly, bin_thickness_annual, mbmwea = binned_mb_calc(gdir, modelprms, glacier_rgi_table, fls=fls, glen_a_multiplier=glen_a_mult, fs=fs)
    bin_thick_monthly = get_bin_thick_monthly(bin_massbalclim_monthly, bin_thickness_annual)
    return surf_h_init, bin_thick_monthly, mbmwea


def oib_objective_fx(params, gdir, modelprms, glacier_rgi_table, fls, glen_a, fs, oib_centers, oib_area, oib_diffs, inds_in_oib, error_type='mad', weighted=True, debug=True):
    try:
        surf_h_init, model_monthly_thick, mbmwea = oib_model_fx(params, gdir, modelprms, glacier_rgi_table, fls, glen_a, fs)
        # get time index for 2013
        refidx = np.where(gdir.dates_table.date.values==datetime.datetime(year=2013, month=1, day=1))[0][0]
        model_diffs = model_monthly_thick - model_monthly_thick[:,refidx][:,np.newaxis]
        # only retain time steps where oib data exists
        model_diffs = model_diffs[:,inds_in_oib]
        # aggregate both model and obs to 100 m bins
        nbins = int(np.ceil((oib_centers[-1] - oib_centers[0]) / 100))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            y = np.column_stack([stats.binned_statistic(x=oib_centers, values=x, statistic=np.nanmean, bins=nbins)[0] for x in oib_diffs.T])
            bin_edges = stats.binned_statistic(x=oib_centers, values=oib_diffs[:,0], statistic=np.nanmean, bins=nbins)[1]
            y_pred = np.column_stack([stats.binned_statistic(x=surf_h_init, values=x, statistic=np.nanmean, bins=bin_edges)[0] for x in model_diffs.T])
            weights  = stats.binned_statistic(x=oib_centers, values=oib_area, statistic=np.nanmean, bins=bin_edges)[0]
        y = np.diff(y,axis=1)
        y_pred = np.diff(y_pred,axis=1)
        # get indices where model output and obs are not nan
        mask = [np.logical_and(~np.isnan(a), ~np.isnan(b)) for a, b in zip(y.T, y_pred.T)]
        mask = np.column_stack(mask)
        norm = weights/np.nanmax(weights)
        if not weighted:
            norm[:] = 1 
        norm = np.column_stack([norm for x in range(y.shape[1])])
        if error_type == 'l1':
            mf = np.sum(np.abs(y_pred[mask] - y[mask])*norm[mask])
        elif error_type == 'l2':
            mf = np.sum(((y_pred[mask] - y[mask])*norm[mask]) ** 2)
        elif error_type == 'rmse':
            mf = np.sqrt(np.mean(((y_pred[mask] - y[mask])*norm[mask]) ** 2))
        elif error_type == 'mad':
            mf =  np.mean(np.abs(y_pred[mask] - y[mask])*norm[mask])
        return mf, mbmwea
    except Exception as err:
        if debug:
            print(f'oib_objective_fx() error: {err}')
        return float('inf'), np.nan


def oib_optimize_wrapper(oggmparams_dict, oggmpaths_dict, cfg, initial_guess, gdir, modelprms, glacier_rgi_table, fls, glen_a, fs, oib_centers, oib_area, oib_diffs, inds_in_oib, error_type, weighted, debug):
    # retain oggm config state from main()
    cfg = importlib.import_module(cfg)
    cfg.PARAMS = oggmparams_dict
    cfg.PATHS = oggmpaths_dict
    result = fmin_bfgs(oib_objective_fx, initial_guess, args = (gdir, modelprms, glacier_rgi_table, fls, glen_a, fs, oib_centers, oib_area, oib_diffs, inds_in_oib, error_type, weighted, debug)[0], gtol=1e-2, disp=False)
    # get loss
    loss_value = oib_objective_fx(result, gdir, modelprms, glacier_rgi_table, fls, glen_a, fs, oib_centers, oib_area, oib_diffs, inds_in_oib, error_type, weighted)[0]
    return result, loss_value


def plot_oib_opt(params, gdir, modelprms, glacier_rgi_table, fls, glen_a, fs, oib_centers, oib_area, oib_diffs, inds_in_oib, error_type, weighted, survey_dates, outname=''):
    model_surf_h_init, model_monthly_thick, mbmwea = oib_model_fx(params, gdir, modelprms, glacier_rgi_table, fls, glen_a, fs)
    print(mbmwea,  gdir.mbdata['mb_clim_mwea'] - (2*gdir.mbdata['mb_clim_mwea_err']), gdir.mbdata['mb_clim_mwea'] + (2*gdir.mbdata['mb_clim_mwea_err']))
    # if (mbmwea < gdir.mbdata['mb_clim_mwea'] - (3*gdir.mbdata['mb_clim_mwea_err'])) or (mbmwea > gdir.mbdata['mb_clim_mwea'] + (3*gdir.mbdata['mb_clim_mwea_err'])):
    #     return
    # get misfit between obs and model
    misfit = oib_objective_fx(params, gdir, modelprms, glacier_rgi_table, fls, glen_a, fs, oib_centers, oib_area, oib_diffs, inds_in_oib, error_type, weighted, debug=True)[0]
    # get 2013 model time step index
    refidx = np.where(gdir.dates_table.date.values==datetime.datetime(year=2013, month=1, day=1))[0][0]
    # subtract off from 2013 thickness
    model_diffs = model_monthly_thick - model_monthly_thick[:,refidx][:,np.newaxis]
    # only retain time steps where oib data exists
    model_diffs = model_diffs[:,inds_in_oib]
    # get double diffs
    model_dbl_diffs = np.diff(model_monthly_thick[:,inds_in_oib], axis=1)
    # get oib double diffs
    oib_dbl_diffs = np.diff(oib_diffs, axis=1)

    # plot
    fig,axs = plt.subplots(ncols=2,nrows=2,figsize=(6.5,5), sharex=True, sharey='row', gridspec_kw = {'wspace':0.1, 'hspace':0.1})

    for i,ax in enumerate(axs.flatten()):
        ax.axhline(y=0,c='k',ls=':',lw=1)
        tax = ax.twinx()
        tax.fill_between(oib_centers, 0, np.asarray(oib_area)*1e-6, color='gray', alpha=.125)
        tax.set_ylim([0,tax.get_ylim()[1]])
        tax.yaxis.set_label_position('right')
        tax.yaxis.set_ticks_position('right')
        tax.spines['right'].set_color('gray')
        tax.yaxis.label.set_color('gray')
        tax.tick_params(axis='y', colors='gray')
        tax.spines['right'].set_color('gray')
        if i%2==0:
            tax.set_yticklabels([])
    
    cmlist = cmocean.cm.matter_r(np.linspace(0,1, oib_diffs.shape[1]))
    for i in range(oib_diffs.shape[1]):
        try:
            axs[0,0].plot(oib_centers,oib_diffs[:,i],c=cmlist[i])
        except:
            pass
        dt = survey_dates.iloc[i]
        axs[0,1].plot(model_surf_h_init, model_diffs[:,i], c=cmlist[i], label=str(dt.year)+"-"+str(dt.month))
    axs[0,1].legend(fontsize=7, loc='upper right', handlelength=1, borderaxespad=0)
    axs[0,0].set_title('NASA OIB',size=10)
    axs[0,1].set_title('PyGEM',size=10)
    axs[0,0].set_xlim([model_surf_h_init.min(), model_surf_h_init.max()])

    cmlist = cmocean.cm.matter_r(np.linspace(0,1, model_dbl_diffs.shape[1]))
    for i in range(model_dbl_diffs.shape[1]):
        try:
            axs[1,0].plot(oib_centers,oib_dbl_diffs[:,i],c=cmlist[i])
        except:
            pass
        dt0 = survey_dates.iloc[i]
        dt1 = survey_dates.iloc[i+1]
        axs[1,1].plot(model_surf_h_init,model_dbl_diffs[:,i], c=cmlist[i], label=f'{dt0.year}-{dt0.month} -- {dt1.year}-{dt1.month}')
    axs[1,1].legend(fontsize=7, loc='upper right', handlelength=1, borderaxespad=0)
    axs[1,0].text(0.5, .05, 'elevation (m)', size=10, horizontalalignment='center', 
                    verticalalignment='center', transform=fig.transFigure)
    axs[1,0].text(.05, .5, 'elevation change (m)', size=10, horizontalalignment='center', 
                    verticalalignment='center', rotation=90, transform=fig.transFigure)
    axs[1,0].text(.95, .5, 'area (km$^2$)', size=10, horizontalalignment='center', color='gray',
                    verticalalignment='center', rotation=90, transform=fig.transFigure)
    glacierstr = gdir.rgi_id.split('-')[1]
    axs[0,0].text(0.5, .95, f'{glacierstr}\nkp:{np.round(params[0],2)}, tbias:{np.round(params[1],2)}, ddfsnow:{np.round(params[2],4)}, mbmwea:{np.round(mbmwea,2)}, {error_type}:{np.round(misfit,2)}', size=10, horizontalalignment='center', 
                    verticalalignment='center', transform=fig.transFigure)
    
    fig.tight_layout()

    if outname:
        fig.savefig(f'{outname}.png')
    plt.show()
    # plt.show(block=False)
    # plt.pause(2)  # Pause for 2 seconds
    # plt.close()
    # return


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

    if args.debug == 1:
        debug = True
    else:
        debug = False

    # ===== LOAD GLACIERS =====
    main_glac_rgi = modelsetup.selectglaciersrgitable(glac_no=glac_no)

    # ===== TIME PERIOD =====
    dates_table = modelsetup.datesmodelrun(
            startyear=pygem_prms.ref_startyear, endyear=pygem_prms.ref_endyear, spinupyears=pygem_prms.ref_spinupyears,
            option_wateryear=pygem_prms.ref_wateryear)
    if debug:
        print(f'model time period:\n{dates_table}')

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
        # for batman in [0]:  
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
                
                    
            except Exception as err:
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

        # ----- CALIBRATION OPTIONS ------
        if (fls is not None) and (gdir.mbdata is not None) and (glacier_area.sum() > 0):
            
            modelprms = {'kp': pygem_prms.kp,
                        'tbias': pygem_prms.tbias,
                        'ddfsnow': pygem_prms.ddfsnow,
                        'ddfice': pygem_prms.ddfice,
                        'tsnow_threshold': pygem_prms.tsnow_threshold,
                        'precgrad': pygem_prms.precgrad}
            
            #%% ===== GRID SEARCH =====
            if pygem_prms.option_calibration == 'grid_search':
                if pygem_prms.opt_calib_monthly_thick:
                    # get rgi7id to load oib data
                    rgi7id = get_rgi7id(glacier_str, debug=debug)
                    oib_dict = load_oib(rgi7id)
                    # get oib diffs
                    oib_centers, oib_area, oib_diffs, oib_dates = get_oib_diffs(oib_dict=oib_dict, debug=debug)
                    # only retain diffs for survey dates where we have pygem data
                    oib_inds_in_pygem = np.intersect1d(oib_dates.to_numpy(), gdir.dates_table.date.to_numpy(), return_indices=True)[1]
                    oib_diffs = oib_diffs[:,oib_inds_in_pygem]
                    oib_dates = oib_dates[oib_inds_in_pygem]
                    # get pygem time step indices where we have oib data
                    inds_in_oib = np.intersect1d(gdir.dates_table.date.to_numpy(), oib_dates.to_numpy(), return_indices=True)[1]

                    # get glen_a
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
                    # perform gradient descent to get pygem model parameters
                    steps = 8
                    # Generate parameter combinations
                    kp_grid, tbias_grid, ddfsnow_grid = np.meshgrid(np.linspace(1, 5, steps),
                                                                    np.linspace(-5, 5, steps),
                                                                    np.linspace(.001, .01, steps),
                                                                    indexing='ij')

                    # Reshape the grids into flat arrays
                    kp_grid = kp_grid.flatten()
                    tbias_grid = tbias_grid.flatten()
                    ddfsnow_grid = ddfsnow_grid.flatten()

                    # Form tuples of parameter combinations
                    param_combinations = list(zip(kp_grid, tbias_grid, ddfsnow_grid))

                    kps = []
                    tbs = []
                    ddfs = []
                    mfs = []
                    mbs = []

                    best_score = float('inf')  # initialize with a high value
                    best_params = None
                    t0 = time.time()
                    # Loop through each parameter combination
                    for param_tuple in param_combinations:
                        mf, mbmwea = oib_objective_fx(param_tuple, gdir, modelprms, glacier_rgi_table, fls, glen_a_multiplier, fs, oib_centers, oib_area, oib_diffs, inds_in_oib, args.errortype, args.weighted, debug)
                        kps.append(param_tuple[0])
                        tbs.append(param_tuple[1])
                        ddfs.append(param_tuple[2])
                        mbs.append(mbmwea)
                        mfs.append(mf)

                        # Check if the current score is better than the best found so far
                        if mf < best_score:
                            best_score = mf
                            best_params = param_tuple

                    print('Grid search parameter optimization time:', time.time()-t0, 's')
                    print("Best Parameters:", best_params)
                    print("Best Score:", best_score)
                    results = pd.DataFrame({'kp':kps,
                                            'tbias':tbs,
                                            'ddfsnow':ddfs,
                                            'mbmwea':mbs,
                                            'misfit':mfs})
                    if args.weighted:
                        outname=glacier_str+f'_gridsearch_w{args.errortype}'
                    else:
                        outname=glacier_str+f'_gridsearch_{args.errortype}'

                    os.makedirs(pygem_prms.output_filepath + '/calibration/' + glacier_str.split('.')[0].zfill(2), exist_ok=True)
                    results.to_csv(pygem_prms.output_filepath + '/calibration/' + glacier_str.split('.')[0].zfill(2) + '/' + outname + '.csv', index=False)

                    if debug:
                        plot_oib_opt(best_params,
                                gdir=gdir, 
                                modelprms=modelprms, 
                                glacier_rgi_table=glacier_rgi_table, 
                                fls=fls, 
                                glen_a=glen_a_multiplier, 
                                fs=fs, 
                                oib_centers=oib_centers, 
                                oib_area=oib_area,
                                oib_diffs=oib_diffs, 
                                inds_in_oib=inds_in_oib,
                                error_type=args.errortype,
                                weighted=args.weighted,
                                survey_dates=oib_dates,
                                outname=outname)


            #%% ===== GRADIENT DESCENT =====
            if pygem_prms.option_calibration == 'gradient_descent':
                if pygem_prms.opt_calib_monthly_thick:
                    # get rgi7id to load oib data
                    rgi7id = get_rgi7id(glacier_str, debug=debug)
                    oib_dict = load_oib(rgi7id)
                    # get oib diffs
                    oib_centers, oib_area, oib_diffs, oib_dates = get_oib_diffs(oib_dict=oib_dict, debug=debug)
                    # only retain diffs for survey dates where we have pygem data
                    oib_inds_in_pygem = np.intersect1d(oib_dates.to_numpy(), gdir.dates_table.date.to_numpy(), return_indices=True)[1]
                    oib_diffs = oib_diffs[:,oib_inds_in_pygem]
                    oib_dates = oib_dates[oib_inds_in_pygem]
                    # get pygem time step indices where we have oib data
                    inds_in_oib = np.intersect1d(gdir.dates_table.date.to_numpy(), oib_dates.to_numpy(), return_indices=True)[1]

                    # get glen_a
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
                    # perform gradient descent to get pygem model parameters
                    steps = 5
                    param_ranges = [
                        np.linspace(1, 5, steps),       # Range for kp
                        np.linspace(-5, 5, steps),      # Range for tbias
                        np.linspace(.001, .01, steps)   # Range for parameter ddfsnow
                    ]
                    # create a list of all parameter combinations to test
                    param_grid = [(x, y, z) for x in param_ranges[0] for y in param_ranges[1] for z in param_ranges[2]]

                    # use multiprocessing.Pool to distribute tasks
                    with multiprocessing.Pool(processes=6) as pool:
                        t0 = time.time()
                        if args.kp and args.tbias and args.ddfsnow:
                            best_params = [args.kp, args.tbias, args.ddfsnow]
                            best_loss= None
                        else:
                            results = pool.map(
                                partial(oib_optimize_wrapper, cfg.PARAMS, cfg.PATHS, 'oggm.cfg',
                                        gdir=gdir, 
                                        modelprms=modelprms, 
                                        glacier_rgi_table=glacier_rgi_table, 
                                        fls=fls, 
                                        glen_a=glen_a_multiplier, 
                                        fs=fs, 
                                        oib_centers=oib_centers, 
                                        oib_area=oib_area,
                                        oib_diffs=oib_diffs, 
                                        inds_in_oib=inds_in_oib, 
                                        error_type=args.errortype, 
                                        weighted=args.weighted,
                                        debug=debug),
                                        param_grid)
                            print(results)
                            # find the best result from all optimizations
                            best_params, best_loss = min(results, key=lambda x: x[1])
                        if debug:
                            print('parameter optimization processing time:', time.time()-t0, 's')
                            print(f'optimized parameters: {best_params}')
                            print(f'misfit: {best_loss}')
                            plot_oib_opt(best_params,
                                    gdir=gdir, 
                                    modelprms=modelprms, 
                                    glacier_rgi_table=glacier_rgi_table, 
                                    fls=fls, 
                                    glen_a=glen_a_multiplier, 
                                    fs=fs, 
                                    oib_centers=oib_centers, 
                                    oib_area=oib_area,
                                    oib_diffs=oib_diffs, 
                                    inds_in_oib=inds_in_oib,
                                    error_type=args.errortype,
                                    weighted=args.weighted,
                                    survey_dates=oib_dates)


            #%% ===== EMULATOR TO SETUP MCMC ANALYSIS AND/OR RUN HH2015 WITH EMULATOR =====
            # - precipitation factor, temperature bias, degree-day factor of snow
            if pygem_prms.option_calibration == 'emulator':
                tbias_step = pygem_prms.tbias_step
                tbias_init = pygem_prms.tbias_init
                kp_init = pygem_prms.kp_init
                ddfsnow_init = pygem_prms.ddfsnow_init
                if pygem_prms.opt_calib_monthly_thick:
                    y_cn = 'bin_thick_monthly'
                else:
                    y_cn = 'mb_mwea'
                
                # ----- Initialize model parameters -----
                modelprms['tbias'] = tbias_init
                modelprms['kp'] = kp_init
                modelprms['ddfsnow'] = ddfsnow_init
                modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                
                # Load sims df
                sims_fp = pygem_prms.emulator_fp + 'sims/' + glacier_str.split('.')[0].zfill(2) + '/'
                sims_fn = glacier_str + '-' + str(pygem_prms.emulator_sims) + '_emulator_sims.json'

                if not os.path.exists(sims_fp + sims_fn) or pygem_prms.overwrite_em_sims:
                    # instantiate dictionary to hold output arrays
                    sims_dict = {}

                    # ----- Temperature bias bounds (ensure reasonable values) -----
                    # Tbias lower bound based on some bins having negative climatic mass balance
                    tbias_maxacc = (-1 * (gdir.historical_climate['temp'] + gdir.historical_climate['lr'] *
                                    (fls[0].surface_h.min() - gdir.historical_climate['elev'])).max())
                    modelprms['tbias'] = tbias_maxacc
                    nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    while nbinyears_negmbclim < 10 or mb_mwea > mb_obs_mwea:
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        if debug:
                            print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3),
                                  'nbinyears_negmbclim:', nbinyears_negmbclim)        
                    tbias_stepsmall = 0.05
                    while nbinyears_negmbclim > 10:
                        modelprms['tbias'] = modelprms['tbias'] - tbias_stepsmall
                        nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        if debug:
                            print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3),
                                  'nbinyears_negmbclim:', nbinyears_negmbclim)
                    # Tbias lower bound 
                    tbias_bndlow = modelprms['tbias'] + tbias_stepsmall
                    modelprms['tbias'] = tbias_bndlow
                    nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    sims_dict = dict_append(
                                            dictionary = sims_dict,
                                            keys=['tbias','kp','ddfsnow','mb_mwea','nbinyears_negmbclim'],
                                            vals = [modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], mb_mwea, nbinyears_negmbclim]
                                            )
                    
                    # Tbias lower bound & high precipitation factor
                    modelprms['kp'] = stats.gamma.ppf(0.99, pygem_prms.kp_gamma_alpha, scale=1/pygem_prms.kp_gamma_beta)
                    nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                    sims_dict = dict_append(
                                            dictionary = sims_dict,
                                            keys=['tbias','kp','ddfsnow','mb_mwea','nbinyears_negmbclim'],
                                            vals = [modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], mb_mwea, nbinyears_negmbclim]
                                            )
                    
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
                        nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        sims_dict = dict_append(
                                                dictionary = sims_dict,
                                                keys=['tbias','kp','ddfsnow','mb_mwea','nbinyears_negmbclim'],
                                                vals = [modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], mb_mwea, nbinyears_negmbclim]
                                                )

                        tbias_middle = modelprms['tbias'] - tbias_step / 2
                        ncount_tbias += 1
                        if debug:
                            print(ncount_tbias, 
                                  'tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))
                    
                    # Tbias upper bound (run for equal amount of steps above the midpoint)
                    while ncount_tbias > 0:
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        sims_dict = dict_append(
                                                dictionary = sims_dict,
                                                keys=['tbias','kp','ddfsnow','mb_mwea','nbinyears_negmbclim'],
                                                vals = [modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], mb_mwea, nbinyears_negmbclim]
                                                )
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
                    
                    # Set up new simulation dictionary if running an emulator for monthly surface elevation change  # FIRST RUN SHOULD BE ON BOUNDS
                    if pygem_prms.opt_calib_monthly_thick:
                        sims_dict = {} 

                        # get oib survey dates for glacier - we'll only emulate glacier thickness/mass balance at these time steps
                        oib_dict = load_oib(get_rgi7id(glacier_str, debug=debug))
                        # get oib diffs
                        oib_dates = get_oib_diffs(oib_dict=oib_dict, debug=debug)[-1]

                        # get glen_a multiplier for ice thickness mass conservation inversion
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

                        # TO-DO: run through predetermined tbias and ddfsnow low and high bounds and add output to sims_dict results

                    # Run through random values
                    for nsim in range(pygem_prms.emulator_sims):
                        print(f'iter {nsim} of {pygem_prms.emulator_sims}')
                        modelprms['tbias'] = tbias_random[nsim]
                        modelprms['kp'] = kp_random[nsim]
                        modelprms['ddfsnow'] = ddfsnow_random[nsim]
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio

                        # create funciton to have monthly binned mass balance
                        # output array will be time, elevation (of bin), model parameters, mass balance at that bin
                        # Option get monthly bin thickness
                        if pygem_prms.opt_calib_monthly_thick:
                            try:
                                surf_h_init, bin_massbalclim_monthly, bin_thickness_annual, _ = binned_mb_calc(gdir, modelprms, glacier_rgi_table, fls=fls, glen_a_multiplier=glen_a_multiplier, fs=fs)
                            except:
                                continue
                            # calculate binned monthly ice thickness - ravel so that output dimension is len(r*c), where r is the number of time steps and c is the number of elevaiton bins
                            bin_thick_monthly = get_bin_thick_monthly(bin_massbalclim_monthly, bin_thickness_annual)
                            # only retain bin_mbtot_monthly where we have oib data for reducing computational cost of emulator
                            oib_dates_idx = np.intersect1d(gdir.dates_table.date.to_numpy(), oib_dates.to_numpy(),return_indices=True)[1]
                            bin_thick_monthly = bin_thick_monthly[:,oib_dates_idx]

                            # only retain bins with nonzero thickness
                            nonzero_rows = np.all(bin_thick_monthly != 0, axis=1)
                            bin_thick_monthly = bin_thick_monthly[nonzero_rows,:]
                            surf_h_init = surf_h_init[nonzero_rows]

                            # need to sparsify this emulator training data - could randonly select, or perhaps select from upper and lower 25th percentiles of glacier
                            # select random bins
                            # take 3 to 5 bins  (minimum, max, 25th, 75th, median) - test performance (3 v 5 v 10, etc)
                            idxs_keep = [index_closest(surf_h_init, np.percentile([np.min(surf_h_init),np.max(surf_h_init)], val)) for val in [0, 25, 50, 75, 100]]

                            # randomly sample elevation bins and see how emulator performs
                            idxs_keep = random.sample(np.arange(len(surf_h_init)).tolist(), 5)
                            
                            # idxs_keep = random.sample(np.arange(bin_thick_monthly.shape[0]).tolist(),bin_thick_monthly.shape[0]//10)
                            bin_thick_monthly = bin_thick_monthly[idxs_keep,:]
                            surf_h_init = surf_h_init[idxs_keep]
                            nbins,nsteps = bin_thick_monthly.shape

                            bin_thick_monthly = np.ravel(bin_thick_monthly)
                            # update sims_dict - we'll need to repeat parameters nxm times (where n is the number of elevation bins, m is the number of time steps) - dataset will be of length (nbins*nsteps*nsims), minus any sparsification
                            sims_dict = dict_append(
                                                    dictionary = sims_dict,
                                                    keys =   ['tbias','kp','ddfsnow','time','bin_h_init','bin_thick_monthly'],
                                                    vals =  [
                                                            np.repeat(modelprms['tbias'], len(bin_thick_monthly)).tolist(), 
                                                            np.repeat(modelprms['kp'], len(bin_thick_monthly)).tolist(), 
                                                            np.repeat(modelprms['ddfsnow'], len(bin_thick_monthly)).tolist(), 
                                                            np.repeat(oib_dates_idx, nbins).tolist(),
                                                            np.repeat(surf_h_init, nsteps).tolist(),
                                                            bin_thick_monthly.tolist()
                                                            ]
                                                    )
                        
                            if nsim==0 and debug:
                                print(pd.DataFrame(sims_dict).head())
                                print(pd.DataFrame(sims_dict).tail())

                        else:
                            # number of bins that have negative clim. mass balance for a given year, mass balance for 2000-2019 period, optionally return binned monthly climatic mass balance
                            nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                            # append results to emulator output dict
                            sims_dict = dict_append(
                                                    dictionary = sims_dict,
                                                    keys=['tbias','kp','ddfsnow','mb_mwea','nbinyears_negmbclim'],
                                                    vals = [modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], mb_mwea, nbinyears_negmbclim]
                                                    )

                        if debug and nsim%500 == 0:
                            print(nsim, 'tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                  'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))

                    # ----- Export results -----
                    # if pygem_prms.opt_calib_monthly_thick:
                    #     sims_fn = sims_fn[:-5]+'_dzdt.json'
                    if os.path.exists(sims_fp) == False:
                        os.makedirs(sims_fp, exist_ok=True)
                    with open(sims_fp + sims_fn, 'w') as fp:
                        json.dump(sims_dict, fp, default=str)
                
                else:
                    # Load simulations
                    with open(sims_fp + sims_fn, 'r') as fp:
                        sims_dict = json.load(fp)

                # ----- EMULATOR: Mass balance -----
                em_mod_fn = glacier_str + f'-emulator-{y_cn}.pth'
                em_mod_fp = pygem_prms.emulator_fp + 'models/' + glacier_str.split('.')[0].zfill(2) + '/'
                if not os.path.exists(em_mod_fp + em_mod_fn) or pygem_prms.overwrite_em_sims:
                    (X_train, X_mean, X_std, y_train, y_mean, y_std, likelihood, model) = (
                            create_emulator(glacier_str, sims_dict, y_cn=y_cn, debug=debug))
                else:
                    # ----- LOAD EMULATOR -----
                    # This is required for the supercomputer such that resources aren't stolen from other cpus
#                    torch.set_num_threads = 1
                    torch.set_num_threads(1)
    
                    state_dict = torch.load(em_mod_fp + em_mod_fn)
                    
                    emulator_extra_fn = em_mod_fn.replace('.pth','_extra.pkl')
                    with open(em_mod_fp + emulator_extra_fn, 'rb') as f:
                        emulator_extra_dict = pickle.load(f)
                        
                    X_train = emulator_extra_dict['X_train']
                    X_mean = emulator_extra_dict['X_mean']
                    X_std = emulator_extra_dict['X_std']
                    y_train = emulator_extra_dict['y_train']
                    y_mean = emulator_extra_dict['y_mean']
                    y_std = emulator_extra_dict['y_std']
                    
                    # initialize likelihood and model
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    
                    # Create a new GP model
                    model = ExactGPModel(X_train, y_train, likelihood)  
                    model.load_state_dict(state_dict)
                    model.eval()
                    

                # ===== HH2015 MODIFIED CALIBRATION USING EMULATOR =====
                if pygem_prms.opt_hh2015_mod:
                    tbias_init = pygem_prms.tbias_init
                    tbias_step = pygem_prms.tbias_step
                    kp_init = pygem_prms.kp_init
                    kp_bndlow = pygem_prms.kp_bndlow
                    kp_bndhigh = pygem_prms.kp_bndhigh
                    ddfsnow_init = pygem_prms.ddfsnow_init

                    # ----- FUNCTIONS -----
                    def run_emulator_mb(modelprms):
                        """ Run the emulator
                        """
                        modelprms_1d_norm = ((np.array([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']]) - 
                                              X_mean) / X_std)                    
                        modelprms_2d_norm = modelprms_1d_norm.reshape(1,3)
                        mb_mwea_norm = model(torch.tensor(modelprms_2d_norm).to(torch.float)).mean.detach().numpy()[0]
                        mb_mwea = mb_mwea_norm * y_std + y_mean
                        return mb_mwea
                    

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
                        mb_mwea_mid_new = run_emulator_mb(modelprms)
    
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
                        mb_mwea_low = run_emulator_mb(modelprms)
                        # Upper bound
                        modelprms[prm2opt] = prm_bndhigh
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                        mb_mwea_high = run_emulator_mb(modelprms)
                        # Middle bound
                        prm_mid = (prm_bndlow + prm_bndhigh) / 2
                        modelprms[prm2opt] = prm_mid
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                        mb_mwea_mid = run_emulator_mb(modelprms)
                        
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
                    sims_df = pd.DataFrame(sims_dict)
                    if debug:
                        sims_df['mb_em'] = np.nan
                        for nidx in sims_df.index.values:
                            modelprms['tbias'] = sims_df.loc[nidx,'tbias']
                            modelprms['kp'] = sims_df.loc[nidx,'kp']
                            modelprms['ddfsnow'] = sims_df.loc[nidx,'ddfsnow']
                            sims_df.loc[nidx,'mb_em'] = run_emulator_mb(modelprms)
                        sims_df['mb_em_dif'] = sims_df['mb_em'] - sims_df['mb_mwea'] 
                    
                    # ----- TEMPERATURE BIAS BOUNDS -----
                    # Selects from emulator sims dataframe
                    sims_df_subset = sims_df.loc[sims_df['kp']==1, :]
                    tbias_bndhigh = sims_df_subset['tbias'].max()
                    tbias_bndlow = sims_df_subset['tbias'].min()
                    
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
                    mb_mwea_bndhigh = run_emulator_mb(modelprms)
                    # Lower bound
                    modelprms['kp'] = kp_bndlow
                    modelprms['tbias'] = tbias_bndhigh
                    modelprms['ddfsnow'] = ddfsnow_init
                    mb_mwea_bndlow = run_emulator_mb(modelprms)
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
                        mb_mwea = run_emulator_mb(modelprms)
                        if mb_mwea > mb_obs_mwea:
                            if debug:
                                print('increase tbias, decrease kp')
                            kp_bndhigh = 1
                            # Check if lower bound causes good agreement
                            modelprms['kp'] = kp_bndlow
                            mb_mwea = run_emulator_mb(modelprms)
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
                                mb_mwea = run_emulator_mb(modelprms)
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
                            mb_mwea = run_emulator_mb(modelprms)
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
                                mb_mwea = run_emulator_mb(modelprms)
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
                        mb_mwea_kp_low = run_emulator_mb(modelprms)
                        # Upper bound
                        modelprms['kp'] = kp_bndhigh
                        mb_mwea_kp_high = run_emulator_mb(modelprms)
                        
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
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[1]
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
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[1]
                    return modelprms, mb_mwea


                # ----- Temperature bias bounds -----
                tbias_bndhigh = 0
                # Tbias lower bound based on no positive temperatures
                tbias_bndlow = (-1 * (gdir.historical_climate['temp'] + gdir.historical_climate['lr'] *
                                (fls[0].surface_h.min() - gdir.historical_climate['elev'])).max())
                modelprms['tbias'] = tbias_bndlow
                mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[1]
                if debug:
                    print('  tbias_bndlow:', np.round(tbias_bndlow,2), 'mb_mwea:', np.round(mb_mwea,2))
                # Tbias upper bound (based on kp_bndhigh)
                modelprms['kp'] = kp_bndhigh
                
                while mb_mwea > mb_obs_mwea and modelprms['tbias'] < 20:
                    modelprms['tbias'] = modelprms['tbias'] + 1
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[1]
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
                mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[1]
                nbinyears_negmbclim = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[0]

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
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[1]
                    while mb_mwea > mb_obs_mwea and test_count < 20:
                        # Update temperature bias
                        modelprms['tbias'] = modelprms['tbias'] + tbias_step
                        # Update bounds
                        tbias_bndhigh_opt = modelprms['tbias']
                        tbias_bndlow_opt = modelprms['tbias'] - tbias_step
                        # Compute mass balance
                        mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[1]
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
                    mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[1]
                    
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
                        mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)[1]
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
                text_file.write(glacier_str + ' had no flowlines or mb_data.')

    # Global variables for Spyder development
    if args.option_parallels == 0:
        global main_vars
        main_vars = inspect.currentframe().f_locals


#%% PARALLEL PROCESSING
if __name__ == '__main__':
    time_start = time.time()
    parser = getparser()
    args = parser.parse_args()

    if args.debug:
        debug = True
    else:
        debug = False

#    cfg.initialize()
#    cfg.PARAMS['use_multiprocessing']  = False
#    if not 'pygem_modelprms' in cfg.BASENAMES:
#        cfg.BASENAMES['pygem_modelprms'] = ('pygem_modelprms.pkl', 'PyGEM model parameters')

    # RGI glacier number
    if args.rgi_glac_number_fn is not None:
        with open(args.rgi_glac_number_fn, 'rb') as f:
            glac_no = pickle.load(f)
    elif pygem_prms.glac_no is not None:
        glac_no = pygem_prms.glac_no
    else:
        main_glac_rgi_all = modelsetup.selectglaciersrgitable(
                rgi_regionsO1=pygem_prms.rgi_regionsO1, rgi_regionsO2=pygem_prms.rgi_regionsO2,
                rgi_glac_number=pygem_prms.rgi_glac_number, include_landterm=pygem_prms.include_landterm,
                include_laketerm=pygem_prms.include_laketerm, include_tidewater=pygem_prms.include_tidewater, 
                min_glac_area_km2=pygem_prms.min_glac_area_km2)
        glac_no = list(main_glac_rgi_all['rgino_str'].values)

    # Number of cores for parallel processing
    if args.option_parallels:
        num_cores = int(np.min([len(glac_no), args.num_simultaneous_processes]))
    else:
        num_cores = 1

    # Glacier number lists to pass for parallel processing
    glac_no_lsts = modelsetup.split_list(glac_no, n=num_cores, option_ordered=args.option_ordered)

    # Read GCM names from argument parser
    gcm_name = args.ref_gcm_name
    print('Processing:', gcm_name)
    
    # Pack variables for multiprocessing
    list_packed_vars = []
    for count, glac_no_lst in enumerate(glac_no_lsts):
        list_packed_vars.append([count, glac_no_lst, gcm_name])

    # Parallel processing
    if args.option_parallels:
        print('Processing in parallel with ' + str(args.num_simultaneous_processes) + ' cores...')
        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
            p.map(main,list_packed_vars)
    # If not in parallel, then only should be one loop
    else:
        # Loop through the chunks and export bias adjustments
        for n in range(len(list_packed_vars)):
            main(list_packed_vars[n])

    print('Total processing time:', time.time()-time_start, 's')