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
import multiprocessing
import os
import glob
import sys
import time
import json
import datetime
# External libraries
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy import stats
#import xarray as xr

# Local libraries
try:
    import pygem
except:
    print('---------\nPyGEM DEV\n---------')
    sys.path.append(os.getcwd() + '/../PyGEM/')
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
#from oggm import graphics
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
    parser.add_argument('-num_simultaneous_processes', action='store', type=int, default=4,
                        help='number of simultaneous processes (cores) to use')
    parser.add_argument('-option_parallels', action='store', type=int, default=1,
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


def binned_mb_calc(gdir, modelprms, glacier_rgi_table, fls=None, glen_a_multiplier=None, fs=None):
    """
    Run the ice thickness inversion and mass balance model to get binned annual ice thickness evolution

    Parameters
    ----------

    Returns
    -------
    out: dict
        dictionary object containing binned initial surface height, monthly climatic mass balance, and annual thickness
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

        # Water Level
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
        if oggm_version > 1.301:
            diag = ev_model.run_until_and_store(nyears)
        else:
            _, diag = ev_model.run_until_and_store(nyears)

        return nfls[0].surface_h, mbmod.glac_bin_massbalclim, mbmod.glac_bin_icethickness_annual


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
    # get annual cumulative climatic mass balance - requires reshaping monthly binned values and summing every 12 months
    bin_massbalclim_annual = bin_massbalclim_monthly.reshape(bin_massbalclim_monthly.shape[0],bin_massbalclim_monthly.shape[1]//12,-1).sum(2)

    # get change in thickness from previous year for each elevation bin
    delta_thick_annual = np.diff(bin_thick_annual, axis=-1)

    # get annual binned flux divergence as annual binned climatic mass balance (-) annual binned ice thickness
    # account for density contrast (convert climatic mass balance in m w.e. to m ice)
    flux_div_annual =   (
                (bin_massbalclim_annual * 
                (pygem_prms.density_ice / pygem_prms.density_water)) - 
                delta_thick_annual
                        )

    # we'll assume the flux divergence is constant througohut the year (is this a good assumption?)
    # ie. take annual values and divide by 12 - use numpy tile to repeat monthly values by 12 months
    flux_div_monthly = np.tile(flux_div_annual / 12, 12)

    # get binned total mass balance (binned climatic mass balance + binned monthly flux divergence)
    # bin_mbtot_monthly = bin_massbalclim_monthly + flux_div_monthly

    # get monthly binned change in thickness assuming constant flux divergence throughout the year
    # account for density contrast (convert monthly climatic mass balance in m w.e. to m ice)
    delta_thick_monthly =   (
                (bin_massbalclim_monthly * 
                (pygem_prms.density_ice / pygem_prms.density_water)) - 
                flux_div_monthly
                            )
    
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
        # This is required for the supercomputer such that resources aren't stolen from other cpus
        torch.set_num_threads(1)
        
        assert y_cn in sims_dict.keys(), 'emulator error: y_cn not in sims_dict'

        ##################        
        ### get X data ###
        ##################
        Xs = []
        # intersection of X variable names with possible options in function declaration
        X_cns = list(set(X_cns).intersection(set(sims_dict.keys())))
        for cn in X_cns:
            Xs.append(sims_dict[cn])

        # convert to numpy arrays
        X = np.column_stack((Xs))
        y = np.array(sims_dict[y_cn])

        # remove any nan's
        nanmask  = ~np.isnan(y)
        X = X[nanmask,:]
        y = y[nanmask]

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
            # kp_1_idx = np.where(sim_dict['kp'] == 1)[0]
            kp_1_idx = np.where(X[:,1] == 1)[0]
            # ax.plot(sims_df.loc[kp_1_idx,'tbias'], sims_df.loc[kp_1_idx,y_cn])
            ax.plot(X[kp_1_idx,0], y[kp_1_idx])
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


def get_oib_dates(rgi7id='', debug=False):
    """
    return Pandas Series object containing Operation IceBridge survey dates (sorted ascendingly)
    """
    oib_fpath = glob.glob(pygem_prms.oib_fp  + f'/diffstats5_*{rgi7id}*.json')
    if len(oib_fpath)==0:
        return
    else:
        oib_fpath = oib_fpath[0]
    # load diffstats file
    with open(oib_fpath, 'rb') as f:
        oib_dict = json.load(f)

    # get sorted list of all month-year datastamps in datafile (add one month to the oib survey dates to be more consistent with PyGEM model timestamps)
    oib_dates = []
    seasons = list(set(oib_dict.keys()).intersection(['march','may','august']))
    for ssn in seasons:
        for yr in list(oib_dict[ssn].keys()):
            doy_int = int(np.ceil(oib_dict[ssn][yr]['mean_doy']))
            dt_obj = datetime.datetime.strptime(f'{int(yr)}-{doy_int}', '%Y-%j')
            oib_dates.append(date_check(dt_obj))
    oib_dates.sort()
    if debug:
        print(f'OIB survey dates:\n{", ".join([str(dt.year)+"-"+str(dt.month)+"-"+str(dt.day) for dt in oib_dates])}')

    return pd.Series(oib_dates)


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


# def load_oib(rgi6id=None, pygem_bin_edges=None, debug=False):

    # loop through survey dates, aggregate to PyGEM bins and get OIB elevation differences from COP30
    # for sd in survey_times:
        # pass

    # edges = np.asarray(oib_diff_dict[season[0]][list(oib_diff_dict[season[0]].keys())[0]]['bin_vals']['bin_start_vec'])
    # edges = np.append(edges,np.asarray(oib_diff_dict[season[0]][list(oib_diff_dict[season[0]].keys())[0]]['bin_vals']['bin_stop_vec'][-1]))
    # centers = (np.asarray(oib_diff_dict[season[0]][list(oib_diff_dict[season[0]].keys())[0]]['bin_vals']['bin_start_vec']) + 
    #             np.asarray(oib_diff_dict[season[0]][list(oib_diff_dict[season[0]].keys())[0]]['bin_vals']['bin_stop_vec'])) / 2

    # bin_area_means, bin_area_edges, _ = scipy.stats.binned_statistic(x=centers, values=cop30_area, statistic=np.nanmean, bins=nbins)

        # # first set zeros to nan so we aren't skewed by no data counts
        # masked=arr.astype(float)
        # masked[masked==0] = np.nan

        # ftype = filt['type']
        # if ftype=='percentage':
        #     if 'threshold' in filt.keys():
        #         pctg = filt['threshold']
        #     else:
        #         pctg=10
        #     mask = arr < np.nanpercentile(masked,pctg)
    
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
                # oib_dat = load_oib(glacier_str, debug=debug)

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
                        oib_dates = get_oib_dates(rgi7id=get_rgi7id(glacier_str, debug=debug), debug=debug)

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
                        print(nsim)
                        modelprms['tbias'] = tbias_random[nsim]
                        modelprms['kp'] = kp_random[nsim]
                        modelprms['ddfsnow'] = ddfsnow_random[nsim]
                        modelprms['ddfice'] = modelprms['ddfsnow'] / pygem_prms.ddfsnow_iceratio

                        # create funciton to have monthly binned mass balance
                        # output array will be time, elevation (of bin), model parameters, mass balance at that bin
                        # Option get monthly bin thickness
                        if pygem_prms.opt_calib_monthly_thick:
                            surf_h_init, bin_massbalclim_monthly, bin_thickness_annual = binned_mb_calc(gdir, modelprms, glacier_rgi_table, fls=fls, glen_a_multiplier=glen_a_multiplier, fs=fs)
                            # calculate binned monthly ice thickness - ravel so that output dimension is len(r*c), where r is the number of time steps and c is the number of elevaiton bins
                            bin_thick_monthly = get_bin_thick_monthly(bin_massbalclim_monthly, bin_thickness_annual)
                            # only retain bin_mbtot_monthly where we have oib data for reducing computational cost of emulator
                            oib_dates_idx = np.intersect1d(gdir.dates_table.date.to_numpy(), oib_dates.to_numpy(),return_indices=True)[1]
                            bin_thick_monthly = bin_thick_monthly[:,oib_dates_idx]
                            nbins,nsteps = bin_thick_monthly.shape

                            bin_thick_monthly = np.ravel(bin_thick_monthly)
                            # update sims_dict - we'll need to repeat parameters nxm times (where n is the number of elevation bins, m is the number of time steps)                        
                            sims_dict = dict_append(
                                                    dictionary = sims_dict,
                                                    keys=   ['tbias','kp','ddfsnow','time','bin_h_init','bin_thick_monthly'],
                                                    vals =  [
                                                            np.repeat(modelprms['tbias'], len(bin_thick_monthly)).tolist(), 
                                                            np.repeat(modelprms['kp'], len(bin_thick_monthly)).tolist(), 
                                                            np.repeat(modelprms['ddfsnow'], len(bin_thick_monthly)).tolist(), 
                                                            np.repeat(oib_dates_idx, nbins).tolist(),
                                                            np.repeat(surf_h_init, nsteps).tolist(),
                                                            bin_thick_monthly.tolist()
                                                            ]
                                                    )
                            # print(sims_dict)
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
                    if pygem_prms.opt_calib_monthly_thick:
                        sims_fn = sims_fn[:-5]+'_dzdt.json'
                    if os.path.exists(sims_fp) == False:
                        os.makedirs(sims_fp, exist_ok=True)
                    with open(sims_fp + sims_fn, 'w') as fp:
                        json.dump(sims_dict, fp, default=str)
                
                else:
                    # Load simulations
                    with open(sims_fp + sims_fn, 'r') as fp:
                        sims_dict = json.load(fp)

                # ----- EMULATOR: Mass balance -----
                em_mod_fn = glacier_str + '-emulator-mb_mwea.pth'
                em_mod_fp = pygem_prms.emulator_fp + 'models/' + glacier_str.split('.')[0].zfill(2) + '/'
                if not os.path.exists(em_mod_fp + em_mod_fn) or pygem_prms.overwrite_em_sims:
                    (X_train, X_mean, X_std, y_train, y_mean, y_std, likelihood, model) = (
                            create_emulator(glacier_str, sims_dict, y_cn='bin_thick_monthly', debug=debug))
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

                    print(sims_dict)
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

    if args.debug == 1:
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
    if args.option_parallels != 0:
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
    if args.option_parallels != 0:
        print('Processing in parallel with ' + str(args.num_simultaneous_processes) + ' cores...')
        with multiprocessing.Pool(args.num_simultaneous_processes) as p:
            p.map(main,list_packed_vars)
    # If not in parallel, then only should be one loop
    else:
        # Loop through the chunks and export bias adjustments
        for n in range(len(list_packed_vars)):
            main(list_packed_vars[n])



    print('Total processing time:', time.time()-time_start, 's')