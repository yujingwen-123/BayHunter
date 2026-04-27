# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import os
import argparse
# set os.environment variables to ensure that numerical computations
# do not do multiprocessing !! Essential !! Do not change !
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import os.path as op
import matplotlib
matplotlib.use('PDF')

from BayHunter import PlotFromStorage
from BayHunter import Targets
from BayHunter import utils
from BayHunter import MCMC_Optimizer
from BayHunter import ModelMatrix
from BayHunter import SynthObs
import logging


#
# console printout formatting
#
formatter = ' %(processName)-12s: %(levelname)-8s |  %(message)s'
logging.basicConfig(format=formatter, level=logging.INFO)
logger = logging.getLogger()


parser = argparse.ArgumentParser(
    description='BayHunter tutorial inversion launcher')
parser.add_argument(
    '--inv', dest='inv_mode', default='joint',
    choices=['joint', 'rf', 'swd'],
    help='Inversion target mode: joint (RF+SWD), rf only, or swd only.')
args = parser.parse_args()


#
# ------------------------------------------------------------  obs SYNTH DATA
#
# Load priors and initparams from config.ini or simply create dictionaries.
initfile = 'config.ini'
priors, initparams = utils.load_params(initfile)

# Load observed data (synthetic test data)
xsw, _ysw = np.loadtxt('observed/st3_rdispph.dat').T
xrf, _yrf = np.loadtxt('observed/st3_prf.dat').T

# add noise to create observed data
# order of noise values (correlation, amplitude):
# noise = [corr1, sigma1, corr2, sigma2] for 2 targets
noise_swd = [0.0, 0.012]
noise_rf = [0.98, 0.005]
ysw_err = SynthObs.compute_expnoise(_ysw, corr=noise_swd[0], sigma=noise_swd[1])
ysw = _ysw + ysw_err
yrf_err = SynthObs.compute_gaussnoise(_yrf, corr=noise_rf[0], sigma=noise_rf[1])
yrf = _yrf + yrf_err
noise = noise_swd + noise_rf


#
# -------------------------------------------  get reference model for BayWatch
#
# Create truemodel only if you wish to have reference values in plots
# and BayWatch. You ONLY need to assign the values in truemodel that you
# wish to have visible.
dep, vs = np.loadtxt('observed/st3_mod.dat', usecols=[0, 2], skiprows=1).T
pdep = np.concatenate((np.repeat(dep, 2)[1:], [150]))
pvs = np.repeat(vs, 2)

selected_targets = []
yobss = []
ymods = []
gauss = []
truenoise = []

if args.inv_mode in ['joint', 'swd']:
    selected_targets.append('swd')
    yobss.append(ysw)
    ymods.append(_ysw)
    gauss.append(False)
    truenoise.extend([noise_swd[0], np.std(ysw_err)])

if args.inv_mode in ['joint', 'rf']:
    selected_targets.append('rf')
    yobss.append(yrf)
    ymods.append(_yrf)
    gauss.append(True)
    truenoise.extend([noise_rf[0], np.std(yrf_err)])

truenoise = np.asarray(truenoise)
explike = SynthObs.compute_explike(
    yobss=yobss, ymods=ymods, noise=truenoise, gauss=gauss,
    rcond=initparams['rcond'])
truemodel = {'model': (pdep, pvs),
             'nlays': 3,
             'noise': truenoise,
             'explike': explike,
             }

print("Inversion mode: %s (%s)" % (args.inv_mode, ','.join(selected_targets)))
print(truenoise, explike)


#
#  -----------------------------------------------------------  DEFINE TARGETS
#
# Only pass x and y observed data to the Targets object which is matching
# the data type. You can chose for SWD any combination of Rayleigh, Love, group
# and phase velocity. Default is the fundamendal mode, but this can be updated.
# For RF chose P or S. You can also use user defined targets or replace the
# forward modeling plugin wih your own module.
targets_list = []
if args.inv_mode in ['joint', 'swd']:
    target_swd = Targets.RayleighDispersionPhase(xsw, ysw, yerr=ysw_err)
    targets_list.append(target_swd)
if args.inv_mode in ['joint', 'rf']:
    target_rf = Targets.PReceiverFunction(xrf, yrf)
    target_rf.moddata.plugin.set_modelparams(gauss=1., water=0.01, p=6.4)
    targets_list.append(target_rf)

# Join the selected targets (one or multiple).
targets = Targets.JointTarget(targets=targets_list)


#
#  ---------------------------------------------------  Quick parameter update
#
# "priors" and "initparams" from config.ini are python dictionaries. You could
# also simply define the dictionaries directly in the script, if you don't want
# to use a config.ini file. Or update the dictionaries as follows, e.g. if you
# have station specific values, etc.
# See docs/bayhunter.pdf for explanation of parameters

priors.update({'mohoest': (38, 4)})  # optional, moho estimate (mean, std)
if args.inv_mode in ['joint', 'rf']:
    priors.update({'rfnoise_corr': noise_rf[0]})
if args.inv_mode in ['joint', 'swd']:
    priors.update({'swdnoise_corr': noise_swd[0]})

initparams.update({'nchains': 5,
                   'iter_burnin': (2048 * 32),
                   'iter_main': (2048 * 16),
                   'propdist': (0.025, 0.025, 0.015, 0.005, 0.005),
                   })


#
#  -------------------------------------------------------  MCMC BAY INVERSION
#
# Save configfile for baywatch. refmodel must not be defined.
utils.save_baywatch_config(targets, path='.', priors=priors,
                           initparams=initparams, refmodel=truemodel)
optimizer = MCMC_Optimizer(targets, initparams=initparams, priors=priors,
                           random_seed=None)
# default for the number of threads is the amount of cpus == one chain per cpu.
# if baywatch is True, inversion data is continuously send out (dtsend)
# to be received by BayWatch (see below).
optimizer.mp_inversion(nthreads=6, baywatch=True, dtsend=1)


#
# #  ---------------------------------------------- Model resaving and plotting
path = initparams['savepath']
cfile = '%s_config.pkl' % initparams['station']
configfile = op.join(path, 'data', cfile)
obj = PlotFromStorage(configfile)
# The final distributions will be saved with save_final_distribution.
# Beforehand, outlier chains will be detected and excluded.
# Outlier chains are defined as chains with a likelihood deviation
# of dev * 100 % from the median posterior likelihood of the best chain.
obj.save_final_distribution(maxmodels=100000, dev=0.05)
# Save a selection of important plots
obj.save_plots(refmodel=truemodel)
obj.merge_pdfs()

#
# If you are only interested on the mean posterior velocity model, type:
# file = op.join(initparams['savepath'], 'data/c_models.npy')
# models = np.load(file)
# singlemodels = ModelMatrix.get_singlemodels(models)
# vs, dep = singlemodels['mean']

#
# #  ---------------------------------------------- WATCH YOUR INVERSION
# if you want to use BayWatch, simply type "baywatch ." in the terminal in the
# folder you saved your baywatch configfile or type the full path instead
# of ".". Type "baywatch --help" for further options.

# if you give your public address as option (default is local address of PC),
# you can also use BayWatch via VPN from 'outside'.
# address = '139.?.?.?'  # here your complete address !!!
