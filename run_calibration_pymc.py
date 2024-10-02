"""Run model calibration."""

# Built-in libraries
import argparse
import inspect
import multiprocessing
import os
import sys
import time
# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import pygem
import pygem_input as pygem_prms
from pygem import class_climate
from pygem.massbalance import PyGEMMassBalance
#from pygem.glacierdynamics import MassRedistributionCurveModel
from pygem.oggm_compat import single_flowline_glacier_directory, single_flowline_glacier_directory_with_calving
import pygem.pygem_modelsetup as modelsetup
from pygem.shop import debris, mbdata, icethickness

#from oggm import cfg
#from oggm import graphics
#from oggm import tasks
#from oggm import utils
from oggm import workflow
#from oggm.core import climate
#from oggm.core.flowline import FluxBasedModel
#from oggm.core.inversion import calving_flux_from_depth

import torch
import gpytorch
import sklearn.model_selection

# Model-specific libraries
if 'MCMC' in pygem_prms.option_calibration:
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
                        help='Filename containing list of rgi_glac_number, helpful for running batches on spc')
    parser.add_argument('-rgi_glac_number', action='store', type=str, default=None,
                        help='rgi glacier number for supercomputer')
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=1,
                        help='number of simultaneous processes (cores) to use (default is 1, ie. no parallelization)')
    # flags
    parser.add_argument('-progress_bar', action='store_true',
                        help='Flag to show progress bar')
    parser.add_argument('-debug', action='store_true',
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

    # evaluate the emulator for a given set of model paramaters
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

        state_dict = torch.load(em_mod_path)
        
        emulator_extra_fp = em_mod_path.replace('.pth', '_extra.pkl')
        with open(emulator_extra_fp, 'rb') as f:
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

    ##################        
    ### get X data ###
    ##################
    
    X = sims_df.loc[:,X_cns].values
    y = sims_df[y_cn].values

    if debug:
        print(f'Calibration x-parameters: {", ".join(X_cns)}')
        print(f'Calibration y-parametes: {y_cn}')
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

        # ----- CALIBRATION OPTIONS ------
        if (fls is not None) and (gdir.mbdata is not None) and (glacier_area.sum() > 0):
            
            modelprms = {'kp': pygem_prms.kp,
                        'tbias': pygem_prms.tbias,
                        'ddfsnow': pygem_prms.ddfsnow,
                        'ddfice': pygem_prms.ddfice,
                        'tsnow_threshold': pygem_prms.tsnow_threshold,
                        'precgrad': pygem_prms.precgrad}
            

            #%% ===== MCMC CALIBRATION ======
            # use MCMC method to determine posterior probability distributions of the three parameters tbias,
            # ddfsnow and kp. Then create an ensemble of parameter sets evenly sampled from these
            # distributions, and output these sets of parameters and their corresponding mass balances to be
            # used in the simulations.
            if pygem_prms.option_calibration == 'MCMC':
                if pygem_prms.option_use_emulator:
                    # ===== Define functions needed for MCMC method =====
                    def run_MCMC(gdir,
                                kp_disttype=pygem_prms.kp_disttype,
                                kp_gamma_alpha=None, kp_gamma_beta=None,
                                kp_lognorm_mu=None, kp_lognorm_tau=None,
                                kp_mu=None, kp_sigma=None, kp_bndlow=None, kp_bndhigh=None,
                                kp_start=None,
                                tbias_disttype=pygem_prms.tbias_disttype,
                                tbias_mu=None, tbias_sigma=None, tbias_bndlow=None, tbias_bndhigh=None,
                                tbias_start=None,
                                ddfsnow_disttype=pygem_prms.ddfsnow_disttype,
                                ddfsnow_mu=pygem_prms.ddfsnow_mu, ddfsnow_sigma=pygem_prms.ddfsnow_sigma,
                                ddfsnow_bndlow=pygem_prms.ddfsnow_bndlow, ddfsnow_bndhigh=pygem_prms.ddfsnow_bndhigh,
                                ddfsnow_start=pygem_prms.ddfsnow_start,
                                iterations=10, mcmc_burn_no=pygem_prms.mcmc_burn_no, thin=pygem_prms.thin_interval, 
                                tune_interval=1000, step=None, tune_throughout=True, save_interval=None, 
                                burn_till_tuned=False, stop_tuning_after=5,
                                verbose=0, progress_bar=args.progress_bar, dbname=None,
                                use_potentials=True, mb_max_loss=None):
                        """
                        Runs the MCMC algorithm.
                        Runs the MCMC algorithm by setting the prior distributions and calibrating the probability
                        distributions of three model parameters for the mass balance function.
                        Parameters
                        ----------
                        kp_disttype : str
                            Distribution type of precipitation factor (either 'lognormal', 'uniform', or 'custom')
                        kp_lognorm_mu, kp_lognorm_tau : float
                            Lognormal mean and tau (1/variance) of precipitation factor
                        kp_mu, kp_sigma, kp_bndlow, kp_bndhigh, kp_start : float
                            Mean, stdev, lower bound, upper bound, and start value of precipitation factor
                        tbias_disttype : str
                            Distribution type of tbias (either 'truncnormal' or 'uniform')
                        tbias_mu, tbias_sigma, tbias_bndlow, tbias_bndhigh, tbias_start : float
                            Mean, stdev, lower bound, upper bound, and start value of temperature bias
                        ddfsnow_disttype : str
                            Distribution type of degree day factor of snow (either 'truncnormal' or 'uniform')
                        ddfsnow_mu, ddfsnow_sigma, ddfsnow_bndlow, ddfsnow_bndhigh, ddfsnow_start : float
                            Mean, stdev, lower bound, upper bound, and start value of degree day factor of snow
                        iterations : int
                            Total number of iterations to do (default 10).
                        mcmc_burn_no : int
                            Variables will not be tallied until this many iterations are complete (default 0).
                        thin : int
                            Variables will be tallied at intervals of this many iterations (default 1).
                        tune_interval : int
                            Step methods will be tuned at intervals of this many iterations (default 1000).
                        step : str
                            Choice of step method to use (default metropolis-hastings).
                        tune_throughout : boolean
                            If true, tuning will continue after the burnin period; otherwise tuning will halt at the end of
                            the burnin period (default True).
                        save_interval : int or None
                            If given, the model state will be saved at intervals of this many iterations (default None).
                        burn_till_tuned: boolean
                            If True the Sampler will burn samples until all step methods are tuned. A tuned step methods is
                            one that was not tuned for the last `stop_tuning_after` tuning intervals. The burn-in phase will
                            have a minimum of 'burn' iterations but could be longer if tuning is needed. After the phase is
                            done the sampler will run for another (iter - burn) iterations, and will tally the samples
                            according to the 'thin' argument. This means that the total number of iteration is updated
                            throughout the sampling procedure.  If True, it also overrides the tune_thorughout argument, so
                            no step method will be tuned when sample are being tallied (default False).
                        stop_tuning_after: int
                            The number of untuned successive tuning interval needed to be reached in order for the burn-in
                            phase to be done (if burn_till_tuned is True) (default 5).
                        verbose : int
                            An integer controlling the verbosity of the models output for debugging (default 0).
                        progress_bar : boolean
                            Display progress bar while sampling (default True).
                        dbname : str
                            Choice of database name the sample should be saved to (default None).
                        use_potentials : Boolean
                            Boolean to use of potential functions to further constrain likelihood functionns
                        mb_max_loss : float
                            Mass balance [mwea] at which the glacier completely melts
                        Returns
                        -------
                        pymc.MCMC.MCMC
                            Returns a model that contains sample traces of tbias, ddfsnow, kp and massbalance. These
                            samples can be accessed by calling the trace attribute. For example:
                                model.trace('ddfsnow')[:]
                            gives the trace of ddfsnow values.
                            A trace, or Markov Chain, is an array of values outputed by the MCMC simulation which defines
                            the posterior probability distribution of the variable at hand.
                        """
                        
                        # ===== EMULATORS FOR FAST PROCESSING =====
                        em_mod_fn = glacier_str + '-emulator-mb_mwea.pth'
                        em_mod_fp = pygem_prms.emulator_fp + 'models/' + glacier_str.split('.')[0].zfill(2) + '/'
                        if not os.path.exists(em_mod_fp + em_mod_fn):
                            mbEmulator = create_emulator(glacier_str, sims_df, y_cn='mb_mwea')
                        else:
                            mbEmulator = massbalEmulator.load(em_mod_path = em_mod_fp + em_mod_fn)
                        
                        
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
                        
                        # ===== CHECK STARTING CONDITIONS (adjust tbias as needed) =====
                        # Test initial model parameters provide good starting condition
                        modelprms['kp'] = kp_start
                        modelprms['tbias'] = tbias_start
                        modelprms['ddfsnow'] = ddfsnow_start
                        
                        # check starting mass balance is not less than the maximum mass loss
                        mb_mwea_start = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                        tbias_step = 0.1
                        while mb_mwea_start < mb_max_loss:
                            modelprms['tbias'] = modelprms['tbias'] - tbias_step
                            mb_mwea_start = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                            
    #                        print('tbias:', modelprms['tbias'], mb_mwea_start)
                            
                        # check melting occurs for starting conditions
                        mb_total_minelev_start = calc_mb_total_minelev(modelprms)
                        tbias_smallstep = 0.01
                        while mb_total_minelev_start > 0 and mb_mwea_start > mb_max_loss:
                            modelprms['tbias'] = modelprms['tbias'] + tbias_smallstep
                            mb_total_minelev_start = calc_mb_total_minelev(modelprms)
                            mb_mwea_start = mbEmulator.eval([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow']])
                            
    #                        print('tbias:', modelprms['tbias'], mb_mwea_start, mb_total_minelev_start)
                        
                        tbias_start = modelprms['tbias']
                        
                        # ===== PRIOR DISTRIBUTIONS =====
                        # Priors dict to record values for export
                        priors_dict = {}
                        priors_dict['kp_disttype'] = kp_disttype
                        priors_dict['tbias_disttype'] = tbias_disttype
                        priors_dict['ddfsnow_disttype'] = ddfsnow_disttype
                        # Precipitation factor [-]
                        if kp_disttype == 'gamma':
                            kp = pymc.Gamma('kp', alpha=kp_gamma_alpha, beta=kp_gamma_beta, value=kp_start)
                            priors_dict['kp_gamma_alpha'] = kp_gamma_alpha
                            priors_dict['kp_gamma_beta'] = kp_gamma_beta
                        elif kp_disttype =='lognormal':
                            #  lognormal distribution (roughly 0.3 to 3)
                            kp_start = np.exp(kp_start)
                            kp = pymc.Lognormal('kp', mu=kp_lognorm_mu, tau=kp_lognorm_tau, value=kp_start)
                            priors_dict['kp_lognorm_mu'] = kp_lognorm_mu
                            priors_dict['kp_lognorm_tau'] = kp_lognorm_tau
                        elif kp_disttype == 'uniform':
                            kp = pymc.Uniform('kp', lower=kp_bndlow, upper=kp_bndhigh, value=kp_start)
                            priors_dict['kp_bndlow'] = kp_bndlow
                            priors_dict['kp_bndhigh'] = kp_bndhigh

                        # Temperature bias [degC]
                        if tbias_disttype == 'normal':
                            tbias = pymc.Normal('tbias', mu=tbias_mu, tau=1/(tbias_sigma**2), value=tbias_start)
                            priors_dict['tbias_mu'] = tbias_mu
                            priors_dict['tbias_sigma'] = tbias_sigma
                        elif tbias_disttype =='truncnormal':
                            tbias = pymc.TruncatedNormal('tbias', mu=tbias_mu, tau=1/(tbias_sigma**2),
                                                        a=tbias_bndlow, b=tbias_bndhigh, value=tbias_start)
                            priors_dict['tbias_mu'] = tbias_mu
                            priors_dict['tbias_sigma'] = tbias_sigma
                            priors_dict['tbias_bndlow'] = tbias_bndlow
                            priors_dict['tbias_bndhigh'] = tbias_bndhigh
                        elif tbias_disttype =='uniform':
                            tbias = pymc.Uniform('tbias', lower=tbias_bndlow, upper=tbias_bndhigh, value=tbias_start)
                            priors_dict['tbias_bndlow'] = tbias_bndlow
                            priors_dict['tbias_bndhigh'] = tbias_bndhigh

                        # Degree day factor of snow [mwe degC-1 d-1]
                        #  always truncated normal distribution with mean 0.0041 mwe degC-1 d-1 and standard deviation of
                        #  0.0015 (Braithwaite, 2008), since it's based on data; uniform should only be used for testing
                        if ddfsnow_disttype == 'truncnormal':
                            ddfsnow = pymc.TruncatedNormal('ddfsnow', mu=ddfsnow_mu, tau=1/(ddfsnow_sigma**2),
                                                        a=ddfsnow_bndlow, b=ddfsnow_bndhigh, value=ddfsnow_start)
                            priors_dict['ddfsnow_mu'] = ddfsnow_mu
                            priors_dict['ddfsnow_sigma'] = ddfsnow_sigma
                            priors_dict['ddfsnow_bndlow'] = ddfsnow_bndlow
                            priors_dict['ddfsnow_bndhigh'] = ddfsnow_bndhigh
                        elif ddfsnow_disttype == 'uniform':
                            ddfsnow = pymc.Uniform('ddfsnow', lower=ddfsnow_bndlow, upper=ddfsnow_bndhigh,
                                                value=ddfsnow_start)
                            priors_dict['ddfsnow_bndlow'] = ddfsnow_bndlow
                            priors_dict['ddfsnow_bndhigh'] = ddfsnow_bndhigh

                        # ===== DETERMINISTIC FUNCTION ====
                        # Define deterministic function for MCMC model based on our a priori probobaility distributions.
                        @deterministic(plot=False)
                        def massbal(tbias=tbias, kp=kp, ddfsnow=ddfsnow):
                            """ Likelihood function for mass balance [mwea] based on model parameters """
                            modelprms_copy = modelprms.copy()
                            if tbias is not None:
                                modelprms_copy['tbias'] = float(tbias)
                            if kp is not None:
                                modelprms_copy['kp'] = float(kp)
                            if ddfsnow is not None:
                                modelprms_copy['ddfsnow'] = float(ddfsnow)
                                modelprms_copy['ddfice'] = modelprms_copy['ddfsnow'] / pygem_prms.ddfsnow_iceratio
    #                        mb_mwea = mb_mwea_calc(gdir, modelprms_copy, glacier_rgi_table, fls=fls)
                            mb_mwea = mbEmulator.eval([modelprms_copy['tbias'], modelprms_copy['kp'], modelprms_copy['ddfsnow']])
                            return mb_mwea


                        # ===== POTENTIAL FUNCTIONS =====
                        # Potential functions are used to impose additional constrains on the model
                        @pymc.potential
                        def mb_max(mb_max_loss=mb_max_loss, massbal=massbal):
                            """ Model parameters cannot completely melt the glacier """
                            if massbal < mb_max_loss:
                                return -np.inf
                            else:
                                return 0
                            
                        @pymc.potential
                        def must_melt(tbias=tbias, kp=kp, ddfsnow=ddfsnow):
                            """
                            Likelihood function for mass balance [mwea] based on model parameters
                            """                          
                            modelprms_copy = modelprms.copy()
                            if tbias is not None:
                                modelprms_copy['tbias'] = float(tbias)
                            if kp is not None:
                                modelprms_copy['kp'] = float(kp)
                            if ddfsnow is not None:
                                modelprms_copy['ddfsnow'] = float(ddfsnow)
                                modelprms_copy['ddfice'] = modelprms_copy['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                            mb_total_minelev = calc_mb_total_minelev(modelprms_copy)
                            if mb_total_minelev < 0:
                                return 0
                            else:
                                return -np.inf


                        # ===== OBSERVED DATA =====
                        #  Observed data defines the observed likelihood of mass balances (based on geodetic observations)
                        obs_massbal = pymc.Normal('obs_massbal', mu=massbal, tau=(1/(mb_obs_mwea_err**2)),
                                                value=float(mb_obs_mwea), observed=True)
                        # Set model
                        if use_potentials:
                            model = pymc.MCMC([{'kp':kp, 'tbias':tbias, 'ddfsnow':ddfsnow,
                                            'massbal':massbal, 'obs_massbal':obs_massbal}, mb_max, must_melt])
                        else:
                            model = pymc.MCMC({'kp':kp, 'tbias':tbias, 'ddfsnow':ddfsnow,
                                            'massbal':massbal, 'obs_massbal':obs_massbal})

                        
                        # Step method (if changed from default)
                        #  Adaptive metropolis is supposed to perform block update, i.e., update all model parameters
                        #  together based on their covariance, which would reduce autocorrelation; however, tests show
                        #  doesn't make a difference.
                        if step == 'am':
                            model.use_step_method(pymc.AdaptiveMetropolis, [kp, tbias, ddfsnow], delay = 1000)
                        # Sample
                        if args.progress_bar == 1:
                            progress_bar_switch = True
                        else:
                            progress_bar_switch = False
                        model.sample(iter=iterations, burn=mcmc_burn_no, thin=thin,
                                    tune_interval=tune_interval, tune_throughout=tune_throughout,
                                    save_interval=save_interval, verbose=verbose, progress_bar=progress_bar_switch)
                        # Close database
                        model.db.close()

                        return model, priors_dict
                    
                    try:
                        # ===== RUNNING MCMC =====
                        # Prior distributions (specified or informed by regions)
                        if pygem_prms.priors_reg_fullfn is not None:
                            # Load priors
                            priors_df = pd.read_csv(pygem_prms.priors_reg_fullfn)
                            priors_idx = np.where((priors_df.O1Region == glacier_rgi_table['O1Region']) & 
                                                (priors_df.O2Region == glacier_rgi_table['O2Region']))[0][0]
                            # Precipitation factor priors
                            kp_gamma_alpha = priors_df.loc[priors_idx, 'kp_alpha']
                            kp_gamma_beta = priors_df.loc[priors_idx, 'kp_beta']
                            # Temperature bias priors
                            tbias_mu = priors_df.loc[priors_idx, 'tbias_mean']
                            tbias_sigma = priors_df.loc[priors_idx, 'tbias_std']
                        else:
                            # Precipitation factor priors
                            kp_gamma_alpha = pygem_prms.kp_gamma_alpha
                            kp_gamma_beta = pygem_prms.kp_gamma_beta
                            # Temperature bias priors
                            tbias_mu = pygem_prms.tbias_mu
                            tbias_sigma = pygem_prms.tbias_sigma
                            
                        modelprms_export = {}
                        # fit the MCMC model
                        for n_chain in range(0,pygem_prms.n_chains):
        
                            if debug:
                                print('\n', glacier_str, ' chain' + str(n_chain))
        
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
        
                            # Determine bounds to check TC starting values and estimate maximum mass loss
                            modelprms['kp'] = kp_start
                            modelprms['ddfsnow'] = ddfsnow_start
                            modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                            
                            # ----- TEMPERATURE BIAS BOUNDS -----
                            # Selects from emulator sims dataframe
                            sims_fp = pygem_prms.emulator_fp + 'sims/' + glacier_str.split('.')[0].zfill(2) + '/'
                            sims_fn = glacier_str + '-' + str(pygem_prms.emulator_sims) + '_emulator_sims.csv'
                            sims_df = pd.read_csv(sims_fp + sims_fn)
                            sims_df_subset = sims_df.loc[sims_df['kp']==1, :]
                            tbias_bndhigh = sims_df_subset['tbias'].max()
                            tbias_bndlow = sims_df_subset['tbias'].min()
                            
                            if debug:
                                print('tbias_bndlow:', np.round(tbias_bndlow,2), 'tbias_bndhigh:', np.round(tbias_bndhigh,2))
                            
                            # Adjust tbias_init based on bounds
                            if tbias_start > tbias_bndhigh:
                                tbias_start = tbias_bndhigh
                            elif tbias_start < tbias_bndlow:
                                tbias_start = tbias_bndlow
                            # ----- Mass balance max loss -----
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
                            
                            
                            if debug:
                                print('\ntbias_start:', np.round(tbias_start,3), 'pf_start:', np.round(kp_start,3),
                                    'ddf_start:', np.round(ddfsnow_start,4), 'mb_max_loss:', np.round(mb_max_loss,2))
                            
                            model, priors_dict = run_MCMC(
                                    gdir,
                                    iterations=pygem_prms.mcmc_sample_no, mcmc_burn_no=pygem_prms.mcmc_burn_no,
                                    step=pygem_prms.mcmc_step,
                                    kp_gamma_alpha=kp_gamma_alpha, kp_gamma_beta=kp_gamma_beta, kp_start=kp_start,
                                    tbias_mu=tbias_mu, tbias_sigma=tbias_sigma, tbias_start=tbias_start,
                                    ddfsnow_start=ddfsnow_start, mb_max_loss=mb_max_loss,
                                    tbias_bndlow=tbias_bndlow, tbias_bndhigh=tbias_bndhigh,
                                    use_potentials=True)                    
                            
                            if debug:
                                print('\nacceptance ratio:', model.step_method_dict[next(iter(model.stochastics))][0].ratio)
                                print('mb_mwea_mean:', np.round(np.mean(model.trace('massbal')[:]),3),
                                    'mb_mwea_std:', np.round(np.std(model.trace('massbal')[:]),3),
                                    '\nmb_obs_mean:', np.round(mb_obs_mwea,3), 'mb_obs_std:', np.round(mb_obs_mwea_err,3))
        
        
                            # Store data from model to be exported
                            chain_str = 'chain_' + str(n_chain)
                            modelprms_export['tbias'] = {chain_str : list(model.trace('tbias')[:])}
                            modelprms_export['kp'] = {chain_str  : list(model.trace('kp')[:])}
                            modelprms_export['ddfsnow'] = {chain_str : list(model.trace('ddfsnow')[:])}
                            modelprms_export['ddfice'] = {chain_str : list(model.trace('ddfsnow')[:] /
                                                                    pygem_prms.ddfsnow_iceratio)}
                            modelprms_export['mb_mwea'] = {chain_str : list(model.trace('massbal')[:])}
        
                        # Export model parameters
                        modelprms_export['precgrad'] = [pygem_prms.precgrad]
                        modelprms_export['tsnow_threshold'] = [pygem_prms.tsnow_threshold]
                        modelprms_export['mb_obs_mwea'] = [mb_obs_mwea]
                        modelprms_export['mb_obs_mwea_err'] = [mb_obs_mwea_err]
                        modelprms_export['priors'] = priors_dict
                        
                        modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                        modelprms_fp = (pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                        + '/')
                        if not os.path.exists(modelprms_fp):
                            os.makedirs(modelprms_fp, exist_ok=True)
                        modelprms_fullfn = modelprms_fp + modelprms_fn
                        if os.path.exists(modelprms_fullfn):
                            with open(modelprms_fullfn, 'rb') as f:
                                modelprms_dict = pickle.load(f)
                            modelprms_dict[pygem_prms.option_calibration] = modelprms_export
                        else:
                            modelprms_dict = {pygem_prms.option_calibration: modelprms_export}
                        with open(modelprms_fullfn, 'wb') as f:
                            pickle.dump(modelprms_dict, f)        
                                
                        if not os.path.exists(modelprms_fp):
                            os.makedirs(modelprms_fp, exist_ok=True)
                        modelprms_fullfn = modelprms_fp + modelprms_fn
                        if os.path.exists(modelprms_fullfn):
                            with open(modelprms_fullfn, 'rb') as f:
                                modelprms_dict = pickle.load(f)
                            modelprms_dict[pygem_prms.option_calibration] = modelprms_export
                        else:
                            modelprms_dict = {pygem_prms.option_calibration: modelprms_export}
                        with open(modelprms_fullfn, 'wb') as f:
                            pickle.dump(modelprms_dict, f)
                        
                        # MCMC LOG SUCCESS
                        mcmc_good_fp = pygem_prms.output_filepath + 'mcmc_success/' + glacier_str.split('.')[0].zfill(2) + '/'
                        if not os.path.exists(mcmc_good_fp):
                            os.makedirs(mcmc_good_fp, exist_ok=True)
                        txt_fn_good = glacier_str + "-mcmc_success.txt"
                        with open(mcmc_good_fp + txt_fn_good, "w") as text_file:
                            text_file.write(glacier_str + ' successfully exported mcmc results')
                    
                    except:
                        # MCMC LOG FAILURE
                        mcmc_fail_fp = pygem_prms.output_filepath + 'mcmc_fail/' + glacier_str.split('.')[0].zfill(2) + '/'
                        if not os.path.exists(mcmc_fail_fp):
                            os.makedirs(mcmc_fail_fp, exist_ok=True)
                        print(mcmc_fail_fp)
                        txt_fn_fail = glacier_str + "-mcmc_fail.txt"
                        with open(mcmc_fail_fp + txt_fn_fail, "w") as text_file:
                            text_file.write(glacier_str + ' failed to complete MCMC')
            
                #%% ===== MCMC FULL SIMULATION CALIBRATION ======
                # Same as MCMC emulator calibration option, but uses full model runs as opposed to an emulator.
                elif not pygem_prms.option_use_emulator:
                    tbias_step = pygem_prms.tbias_step
    #                tbias_init = pygem_prms.tbias_init
                    
                    # ===== Define functions needed for MCMC method =====
                    def run_MCMC(gdir,
                                kp_disttype=pygem_prms.kp_disttype,
                                kp_gamma_alpha=None, kp_gamma_beta=None,
                                kp_lognorm_mu=None, kp_lognorm_tau=None,
                                kp_mu=None, kp_sigma=None, kp_bndlow=None, kp_bndhigh=None,
                                kp_start=None,
                                tbias_disttype=pygem_prms.tbias_disttype,
                                tbias_mu=None, tbias_sigma=None, tbias_bndlow=None, tbias_bndhigh=None,
                                tbias_start=None,
                                ddfsnow_disttype=pygem_prms.ddfsnow_disttype,
                                ddfsnow_mu=pygem_prms.ddfsnow_mu, ddfsnow_sigma=pygem_prms.ddfsnow_sigma,
                                ddfsnow_bndlow=pygem_prms.ddfsnow_bndlow, ddfsnow_bndhigh=pygem_prms.ddfsnow_bndhigh,
                                ddfsnow_start=pygem_prms.ddfsnow_start,
                                iterations=10, mcmc_burn_no=pygem_prms.mcmc_burn_no, thin=pygem_prms.thin_interval, 
                                tune_interval=1000, step=None, tune_throughout=True, save_interval=None, 
                                burn_till_tuned=False, stop_tuning_after=5,
                                verbose=0, progress_bar=args.progress_bar, dbname=None,
                                use_potentials=True, mb_max_loss=None):
                        """
                        Runs the MCMC algorithm.

                        Runs the MCMC algorithm by setting the prior distributions and calibrating the probability
                        distributions of three model parameters for the mass balance function.

                        Parameters
                        ----------
                        kp_disttype : str
                            Distribution type of precipitation factor (either 'lognormal', 'uniform', or 'custom')
                        kp_lognorm_mu, kp_lognorm_tau : float
                            Lognormal mean and tau (1/variance) of precipitation factor
                        kp_mu, kp_sigma, kp_bndlow, kp_bndhigh, kp_start : float
                            Mean, stdev, lower bound, upper bound, and start value of precipitation factor
                        tbias_disttype : str
                            Distribution type of tbias (either 'truncnormal' or 'uniform')
                        tbias_mu, tbias_sigma, tbias_bndlow, tbias_bndhigh, tbias_start : float
                            Mean, stdev, lower bound, upper bound, and start value of temperature bias
                        ddfsnow_disttype : str
                            Distribution type of degree day factor of snow (either 'truncnormal' or 'uniform')
                        ddfsnow_mu, ddfsnow_sigma, ddfsnow_bndlow, ddfsnow_bndhigh, ddfsnow_start : float
                            Mean, stdev, lower bound, upper bound, and start value of degree day factor of snow
                        iterations : int
                            Total number of iterations to do (default 10).
                        mcmc_burn_no : int
                            Variables will not be tallied until this many iterations are complete (default 0).
                        thin : int
                            Variables will be tallied at intervals of this many iterations (default 1).
                        tune_interval : int
                            Step methods will be tuned at intervals of this many iterations (default 1000).
                        step : str
                            Choice of step method to use (default metropolis-hastings).
                        tune_throughout : boolean
                            If true, tuning will continue after the burnin period; otherwise tuning will halt at the end of
                            the burnin period (default True).
                        save_interval : int or None
                            If given, the model state will be saved at intervals of this many iterations (default None).
                        burn_till_tuned: boolean
                            If True the Sampler will burn samples until all step methods are tuned. A tuned step methods is
                            one that was not tuned for the last `stop_tuning_after` tuning intervals. The burn-in phase will
                            have a minimum of 'burn' iterations but could be longer if tuning is needed. After the phase is
                            done the sampler will run for another (iter - burn) iterations, and will tally the samples
                            according to the 'thin' argument. This means that the total number of iteration is updated
                            throughout the sampling procedure.  If True, it also overrides the tune_thorughout argument, so
                            no step method will be tuned when sample are being tallied (default False).
                        stop_tuning_after: int
                            The number of untuned successive tuning interval needed to be reached in order for the burn-in
                            phase to be done (if burn_till_tuned is True) (default 5).
                        verbose : int
                            An integer controlling the verbosity of the models output for debugging (default 0).
                        progress_bar : boolean
                            Display progress bar while sampling (default True).
                        dbname : str
                            Choice of database name the sample should be saved to (default None).
                        use_potentials : Boolean
                            Boolean to use of potential functions to further constrain likelihood functionns
                        mb_max_loss : float
                            Mass balance [mwea] at which the glacier completely melts

                        Returns
                        -------
                        pymc.MCMC.MCMC
                            Returns a model that contains sample traces of tbias, ddfsnow, kp and massbalance. These
                            samples can be accessed by calling the trace attribute. For example:

                                model.trace('ddfsnow')[:]

                            gives the trace of ddfsnow values.

                            A trace, or Markov Chain, is an array of values outputed by the MCMC simulation which defines
                            the posterior probability distribution of the variable at hand.
                        """
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
                        
                        # ===== CHECK STARTING CONDITIONS (adjust tbias as needed) =====
                        # Test initial model parameters provide good starting condition
                        modelprms['kp'] = kp_start
                        modelprms['tbias'] = tbias_start
                        modelprms['ddfsnow'] = ddfsnow_start
                        
                        # check starting mass balance is not less than the maximum mass loss
    #                    mb_mwea_start = run_emulator_mb(modelprms)
                        mb_mwea_start = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                        tbias_step = 0.1
                        while mb_mwea_start < mb_max_loss:
                            modelprms['tbias'] = modelprms['tbias'] - tbias_step
    #                        mb_mwea_start = run_emulator_mb(modelprms)
                            mb_mwea_start = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                            
                            print('tbias:', modelprms['tbias'], mb_mwea_start)
                            
                        # check melting occurs for starting conditions
                        mb_total_minelev_start = calc_mb_total_minelev(modelprms)
                        tbias_smallstep = 0.01
                        while mb_total_minelev_start > 0 and mb_mwea_start > mb_max_loss:
                            modelprms['tbias'] = modelprms['tbias'] + tbias_smallstep
                            mb_total_minelev_start = calc_mb_total_minelev(modelprms)
    #                        mb_mwea_start = run_emulator_mb(modelprms)
                            mb_mwea_start = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls)
                            
    #                        print('tbias:', modelprms['tbias'], mb_mwea_start, mb_total_minelev_start)
                        
                        tbias_start = modelprms['tbias']
                        
                        # ===== PRIOR DISTRIBUTIONS =====
                        # Priors dict to record values for export
                        priors_dict = {}
                        priors_dict['kp_disttype'] = kp_disttype
                        priors_dict['tbias_disttype'] = tbias_disttype
                        priors_dict['ddfsnow_disttype'] = ddfsnow_disttype
                        # Precipitation factor [-]
                        if kp_disttype == 'gamma':
                            kp = pymc.Gamma('kp', alpha=kp_gamma_alpha, beta=kp_gamma_beta, value=kp_start)
                            priors_dict['kp_gamma_alpha'] = kp_gamma_alpha
                            priors_dict['kp_gamma_beta'] = kp_gamma_beta
                        elif kp_disttype =='lognormal':
                            #  lognormal distribution (roughly 0.3 to 3)
                            kp_start = np.exp(kp_start)
                            kp = pymc.Lognormal('kp', mu=kp_lognorm_mu, tau=kp_lognorm_tau, value=kp_start)
                            priors_dict['kp_lognorm_mu'] = kp_lognorm_mu
                            priors_dict['kp_lognorm_tau'] = kp_lognorm_tau
                        elif kp_disttype == 'uniform':
                            kp = pymc.Uniform('kp', lower=kp_bndlow, upper=kp_bndhigh, value=kp_start)
                            priors_dict['kp_bndlow'] = kp_bndlow
                            priors_dict['kp_bndhigh'] = kp_bndhigh

                        # Temperature bias [degC]
                        if tbias_disttype == 'normal':
                            tbias = pymc.Normal('tbias', mu=tbias_mu, tau=1/(tbias_sigma**2), value=tbias_start)
                            priors_dict['tbias_mu'] = tbias_mu
                            priors_dict['tbias_sigma'] = tbias_sigma
                        elif tbias_disttype =='truncnormal':
                            tbias = pymc.TruncatedNormal('tbias', mu=tbias_mu, tau=1/(tbias_sigma**2),
                                                        a=tbias_bndlow, b=tbias_bndhigh, value=tbias_start)
                            priors_dict['tbias_mu'] = tbias_mu
                            priors_dict['tbias_sigma'] = tbias_sigma
                            priors_dict['tbias_bndlow'] = tbias_bndlow
                            priors_dict['tbias_bndhigh'] = tbias_bndhigh
                        elif tbias_disttype =='uniform':
                            tbias = pymc.Uniform('tbias', lower=tbias_bndlow, upper=tbias_bndhigh, value=tbias_start)
                            priors_dict['tbias_bndlow'] = tbias_bndlow
                            priors_dict['tbias_bndhigh'] = tbias_bndhigh

                        # Degree day factor of snow [mwe degC-1 d-1]
                        #  always truncated normal distribution with mean 0.0041 mwe degC-1 d-1 and standard deviation of
                        #  0.0015 (Braithwaite, 2008), since it's based on data; uniform should only be used for testing
                        if ddfsnow_disttype == 'truncnormal':
                            ddfsnow = pymc.TruncatedNormal('ddfsnow', mu=ddfsnow_mu, tau=1/(ddfsnow_sigma**2),
                                                        a=ddfsnow_bndlow, b=ddfsnow_bndhigh, value=ddfsnow_start)
                            priors_dict['ddfsnow_mu'] = ddfsnow_mu
                            priors_dict['ddfsnow_sigma'] = ddfsnow_sigma
                            priors_dict['ddfsnow_bndlow'] = ddfsnow_bndlow
                            priors_dict['ddfsnow_bndhigh'] = ddfsnow_bndhigh
                        elif ddfsnow_disttype == 'uniform':
                            ddfsnow = pymc.Uniform('ddfsnow', lower=ddfsnow_bndlow, upper=ddfsnow_bndhigh,
                                                value=ddfsnow_start)
                            priors_dict['ddfsnow_bndlow'] = ddfsnow_bndlow
                            priors_dict['ddfsnow_bndhigh'] = ddfsnow_bndhigh

                        # ===== DETERMINISTIC FUNCTION ====
                        # Define deterministic function for MCMC model based on our a priori probobaility distributions.
                        @deterministic(plot=False)
                        def massbal(tbias=tbias, kp=kp, ddfsnow=ddfsnow):
                            """ Likelihood function for mass balance [mwea] based on model parameters """
                            modelprms_copy = modelprms.copy()
                            if tbias is not None:
                                modelprms_copy['tbias'] = float(tbias)
                            if kp is not None:
                                modelprms_copy['kp'] = float(kp)
                            if ddfsnow is not None:
                                modelprms_copy['ddfsnow'] = float(ddfsnow)
                                modelprms_copy['ddfice'] = modelprms_copy['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                            mb_mwea = mb_mwea_calc(gdir, modelprms_copy, glacier_rgi_table, fls=fls)
    #                        mb_mwea = run_emulator_mb(modelprms_copy)
                            return mb_mwea


                        # ===== POTENTIAL FUNCTIONS =====
                        # Potential functions are used to impose additional constrains on the model
                        @pymc.potential
                        def mb_max(mb_max_loss=mb_max_loss, massbal=massbal):
                            """ Model parameters cannot completely melt the glacier """
                            if massbal < mb_max_loss:
                                return -np.inf
                            else:
                                return 0
                            
                        @pymc.potential
                        def must_melt(tbias=tbias, kp=kp, ddfsnow=ddfsnow):
                            """
                            Likelihood function for mass balance [mwea] based on model parameters
                            """                          
                            modelprms_copy = modelprms.copy()
                            if tbias is not None:
                                modelprms_copy['tbias'] = float(tbias)
                            if kp is not None:
                                modelprms_copy['kp'] = float(kp)
                            if ddfsnow is not None:
                                modelprms_copy['ddfsnow'] = float(ddfsnow)
                                modelprms_copy['ddfice'] = modelprms_copy['ddfsnow'] / pygem_prms.ddfsnow_iceratio
                            mb_total_minelev = calc_mb_total_minelev(modelprms_copy)
                            if mb_total_minelev < 0:
                                return 0
                            else:
                                return -np.inf


                        # ===== OBSERVED DATA =====
                        #  Observed data defines the observed likelihood of mass balances (based on geodetic observations)
                        obs_massbal = pymc.Normal('obs_massbal', mu=massbal, tau=(1/(mb_obs_mwea_err**2)),
                                                value=float(mb_obs_mwea), observed=True)
                        # Set model
                        if use_potentials:
                            model = pymc.MCMC([{'kp':kp, 'tbias':tbias, 'ddfsnow':ddfsnow,
                                            'massbal':massbal, 'obs_massbal':obs_massbal}, mb_max, must_melt])
                        else:
                            model = pymc.MCMC({'kp':kp, 'tbias':tbias, 'ddfsnow':ddfsnow,
                                            'massbal':massbal, 'obs_massbal':obs_massbal})

                        
                        # Step method (if changed from default)
                        #  Adaptive metropolis is supposed to perform block update, i.e., update all model parameters
                        #  together based on their covariance, which would reduce autocorrelation; however, tests show
                        #  doesn't make a difference.
                        if step == 'am':
                            model.use_step_method(pymc.AdaptiveMetropolis, [kp, tbias, ddfsnow], delay = 1000)
                        # Sample
                        if args.progress_bar == 1:
                            progress_bar_switch = True
                        else:
                            progress_bar_switch = False
                        model.sample(iter=iterations, burn=mcmc_burn_no, thin=thin,
                                    tune_interval=tune_interval, tune_throughout=tune_throughout,
                                    save_interval=save_interval, verbose=verbose, progress_bar=progress_bar_switch)
                        # Close database
                        model.db.close()

                        return model, priors_dict
                        
                    # Record a second version for analysis of full simulations
                    modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                    modelprms_fp_fullsim = (pygem_prms.output_filepath + 'calibration-fullsim/' + glacier_str.split('.')[0].zfill(2) 
                                    + '/')
                    
                    if not os.path.exists(modelprms_fp_fullsim + modelprms_fn):
                        try:
        #                for batman in [0]:
                            # ===== RUNNING MCMC =====
                            # Prior distributions (specified or informed by regions)
                            if pygem_prms.priors_reg_fullfn is not None:
                                # Load priors
                                priors_df = pd.read_csv(pygem_prms.priors_reg_fullfn)
                                priors_idx = np.where((priors_df.O1Region == glacier_rgi_table['O1Region']) & 
                                                    (priors_df.O2Region == glacier_rgi_table['O2Region']))[0][0]
                                # Precipitation factor priors
                                kp_gamma_alpha = priors_df.loc[priors_idx, 'kp_alpha']
                                kp_gamma_beta = priors_df.loc[priors_idx, 'kp_beta']
                                # Temperature bias priors
                                tbias_mu = priors_df.loc[priors_idx, 'tbias_mean']
                                tbias_sigma = priors_df.loc[priors_idx, 'tbias_std']
                            else:
                                # Precipitation factor priors
                                kp_gamma_alpha = pygem_prms.kp_gamma_alpha
                                kp_gamma_beta = pygem_prms.kp_gamma_beta
                                # Temperature bias priors
                                tbias_mu = pygem_prms.tbias_mu
                                tbias_sigma = pygem_prms.tbias_sigma
                                
                            modelprms_export = {}
                            # fit the MCMC model
                            for n_chain in range(0,pygem_prms.n_chains):
            
                                if debug:
                                    print('\n', glacier_str, ' chain' + str(n_chain))
            
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
            
                                # Determine bounds to check TC starting values and estimate maximum mass loss
                                modelprms['kp'] = kp_start
                                modelprms['ddfsnow'] = ddfsnow_start
                                modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio
        
                                # ----- TEMPERATURE BIAS BOUNDS -----
                                # ensure reasonable values
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
                                modelprms['kp'] = stats.gamma.ppf(0.99, kp_gamma_alpha, scale=1/kp_gamma_beta)
                                nbinyears_negmbclim, mb_mwea = mb_mwea_calc(gdir, modelprms, glacier_rgi_table, fls=fls,
                                                                            return_tbias_mustmelt_wmb=True)
                                output_single = np.array([modelprms['tbias'], modelprms['kp'], modelprms['ddfsnow'], 
                                                        mb_mwea, nbinyears_negmbclim])
                                output_all = np.vstack((output_all, output_single))
                                
                                if debug:
                                    print('tbias:', np.round(modelprms['tbias'],2), 'kp:', np.round(modelprms['kp'],2),
                                        'ddfsnow:', np.round(modelprms['ddfsnow'],4), 'mb_mwea:', np.round(mb_mwea,3))
            
                                # Tbias 'mid-point'
                                modelprms['kp'] = 1
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
                                        
                                if debug:
                                    print('tbias_bndlow:', np.round(tbias_bndlow,2), 'tbias_bndhigh:', np.round(tbias_bndhigh,2))
                                
                                # Adjust tbias_init based on bounds
                                if tbias_start > tbias_bndhigh:
                                    tbias_start = tbias_bndhigh
                                elif tbias_start < tbias_bndlow:
                                    tbias_start = tbias_bndlow
                                    
                                # ----- Mass balance max loss -----
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
                                
                                
                                if debug:
                                    print('\n',glacier_str, 'tbias_start:', np.round(tbias_start,3), 'pf_start:', np.round(kp_start,3),
                                        'ddf_start:', np.round(ddfsnow_start,4), 'mb_max_loss:', np.round(mb_max_loss,2))
                                
                                model, priors_dict = run_MCMC(
                                        gdir,
                                        iterations=pygem_prms.mcmc_sample_no, mcmc_burn_no=pygem_prms.mcmc_burn_no,
                                        step=pygem_prms.mcmc_step,
                                        kp_gamma_alpha=kp_gamma_alpha, kp_gamma_beta=kp_gamma_beta, kp_start=kp_start,
                                        tbias_mu=tbias_mu, tbias_sigma=tbias_sigma, tbias_start=tbias_start,
                                        ddfsnow_start=ddfsnow_start, mb_max_loss=mb_max_loss,
                                        tbias_bndlow=tbias_bndlow, tbias_bndhigh=tbias_bndhigh,
                                        use_potentials=True)                    
                                
                                if debug:
                                    print('\nacceptance ratio:', model.step_method_dict[next(iter(model.stochastics))][0].ratio)
                                    print('mb_mwea_mean:', np.round(np.mean(model.trace('massbal')[:]),3),
                                        'mb_mwea_std:', np.round(np.std(model.trace('massbal')[:]),3),
                                        '\nmb_obs_mean:', np.round(mb_obs_mwea,3), 'mb_obs_std:', np.round(mb_obs_mwea_err,3))
            
            
                                # Store data from model to be exported
                                chain_str = 'chain_' + str(n_chain)
                                modelprms_export['tbias'] = {chain_str : list(model.trace('tbias')[:])}
                                modelprms_export['kp'] = {chain_str  : list(model.trace('kp')[:])}
                                modelprms_export['ddfsnow'] = {chain_str : list(model.trace('ddfsnow')[:])}
                                modelprms_export['ddfice'] = {chain_str : list(model.trace('ddfsnow')[:] /
                                                                        pygem_prms.ddfsnow_iceratio)}
                                modelprms_export['mb_mwea'] = {chain_str : list(model.trace('massbal')[:])}
            
                            # Export model parameters
                            modelprms_export['precgrad'] = [pygem_prms.precgrad]
                            modelprms_export['tsnow_threshold'] = [pygem_prms.tsnow_threshold]
                            modelprms_export['mb_obs_mwea'] = [mb_obs_mwea]
                            modelprms_export['mb_obs_mwea_err'] = [mb_obs_mwea_err]
                            modelprms_export['priors'] = priors_dict
                            
                            modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                            modelprms_fp = (pygem_prms.output_filepath + 'calibration/' + glacier_str.split('.')[0].zfill(2) 
                                            + '/')
                            if not os.path.exists(modelprms_fp):
                                os.makedirs(modelprms_fp, exist_ok=True)
                            modelprms_fullfn = modelprms_fp + modelprms_fn
                            if os.path.exists(modelprms_fullfn):
                                with open(modelprms_fullfn, 'rb') as f:
                                    modelprms_dict = pickle.load(f)
                                modelprms_dict[pygem_prms.option_calibration] = modelprms_export
                            else:
                                modelprms_dict = {pygem_prms.option_calibration: modelprms_export}
        #                    with open(modelprms_fullfn, 'wb') as f:
        #                        pickle.dump(modelprms_dict, f)        
                                
                            # Record a second version for analysis of full simulations
                            modelprms_fn = glacier_str + '-modelprms_dict.pkl'
                            modelprms_fp_fullsim = (pygem_prms.output_filepath + 'calibration-fullsim/' + glacier_str.split('.')[0].zfill(2) 
                                            + '/')
                            if not os.path.exists(modelprms_fp_fullsim):
                                os.makedirs(modelprms_fp_fullsim, exist_ok=True)
                            with open(modelprms_fp_fullsim + modelprms_fn, 'wb') as f:
                                pickle.dump(modelprms_dict, f)
                            
                            # MCMC LOG SUCCESS
                            mcmc_good_fp = pygem_prms.output_filepath + 'mcmc_success-fullsim/' + glacier_str.split('.')[0].zfill(2) + '/'
                            if not os.path.exists(mcmc_good_fp):
                                os.makedirs(mcmc_good_fp, exist_ok=True)
                            txt_fn_good = glacier_str + "-mcmc_success.txt"
                            with open(mcmc_good_fp + txt_fn_good, "w") as text_file:
                                text_file.write(glacier_str + ' successfully exported mcmc results')
                        
                        except:
                            # MCMC LOG FAILURE
                            mcmc_fail_fp = pygem_prms.output_filepath + 'mcmc_fail-fullsim/' + glacier_str.split('.')[0].zfill(2) + '/'
                            if not os.path.exists(mcmc_fail_fp):
                                os.makedirs(mcmc_fail_fp, exist_ok=True)
                            print(mcmc_fail_fp)
                            txt_fn_fail = glacier_str + "-mcmc_fail.txt"
                            with open(mcmc_fail_fp + txt_fn_fail, "w") as text_file:
                                text_file.write(glacier_str + ' failed to complete MCMC')


        else:
            # LOG FAILURE
            fail_fp = pygem_prms.output_filepath + 'cal_fail/' + glacier_str.split('.')[0].zfill(2) + '/'
            if not os.path.exists(fail_fp):
                os.makedirs(fail_fp, exist_ok=True)
            txt_fn_fail = glacier_str + "-cal_fail.txt"
            with open(fail_fp + txt_fn_fail, "w") as text_file:
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