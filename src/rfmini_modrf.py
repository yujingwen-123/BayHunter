# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import numpy as np

EI = 0 + 1j
SEC_PER_DEG_TO_SEC_PER_KM = 1. / 111.19492664455873


def _e_inverse(omega, rho, alpha, beta, rayp):
    """E_inverse (Aki & Richards, Eq. 5.71)."""
    e_inv = np.zeros((4, 4), dtype=np.complex128)
    eta = np.sqrt(1.0 / (beta * beta) - rayp * rayp)
    xi = np.sqrt(1.0 / (alpha * alpha) - rayp * rayp)
    bp = 1.0 - 2.0 * beta * beta * rayp * rayp

    e_inv[0, 0] = beta * beta * rayp / alpha
    e_inv[0, 1] = bp / (2.0 * alpha * xi)
    e_inv[0, 2] = -rayp / (2.0 * omega * rho * alpha * xi) * EI
    e_inv[0, 3] = -1.0 / (2.0 * omega * rho * alpha) * EI
    e_inv[1, 0] = bp / (2.0 * beta * eta)
    e_inv[1, 1] = -beta * rayp
    e_inv[1, 2] = -1.0 / (2.0 * omega * rho * beta) * EI
    e_inv[1, 3] = rayp / (2.0 * omega * rho * beta * eta) * EI
    e_inv[2, 0] = e_inv[0, 0]
    e_inv[2, 1] = -e_inv[0, 1]
    e_inv[2, 2] = -e_inv[0, 2]
    e_inv[2, 3] = e_inv[0, 3]
    e_inv[3, 0] = e_inv[1, 0]
    e_inv[3, 1] = -e_inv[1, 1]
    e_inv[3, 2] = -e_inv[1, 2]
    e_inv[3, 3] = e_inv[1, 3]
    return e_inv


def _propagator_sol(omega, rho, alpha, beta, rayp, thick):
    """propagator (Aki & Richards, Box 9.1 Eq. 3)."""
    p_mat = np.zeros((4, 4), dtype=np.complex128)
    beta2 = beta * beta
    p2 = rayp * rayp
    bp = 1.0 - 2.0 * beta2 * p2
    eta = np.sqrt(1.0 / beta2 - p2)
    xi = np.sqrt(1.0 / (alpha * alpha) - p2)
    cos_xi = np.cos(omega * xi * thick)
    cos_eta = np.cos(omega * eta * thick)
    sin_xi = np.sin(omega * xi * thick)
    sin_eta = np.sin(omega * eta * thick)

    p_mat[0, 0] = 2.0 * beta2 * p2 * cos_xi + bp * cos_eta
    p_mat[0, 1] = rayp * (bp / xi * sin_xi - 2.0 * beta2 * eta * sin_eta) * EI
    p_mat[0, 2] = (p2 / xi * sin_xi + eta * sin_eta) / (omega * rho)
    p_mat[0, 3] = rayp * (-cos_xi + cos_eta) / (omega * rho) * EI
    p_mat[1, 0] = rayp * (2.0 * beta2 * xi * sin_xi - bp / eta * sin_eta) * EI
    p_mat[1, 1] = bp * cos_xi + 2.0 * beta2 * p2 * cos_eta
    p_mat[1, 2] = p_mat[0, 3]
    p_mat[1, 3] = (xi * sin_xi + p2 / eta * sin_eta) / (omega * rho)
    p_mat[2, 0] = omega * rho * (-4.0 * beta2 * beta2 * p2 * xi * sin_xi -
                                 bp * bp / eta * sin_eta)
    p_mat[2, 1] = 2.0 * omega * beta2 * rho * rayp * bp * (cos_xi - cos_eta) * EI
    p_mat[2, 2] = p_mat[0, 0]
    p_mat[2, 3] = p_mat[1, 0]
    p_mat[3, 0] = p_mat[2, 1]
    p_mat[3, 1] = -omega * rho * (bp * bp / xi * sin_xi +
                                  4.0 * beta2 * beta2 * p2 * eta * sin_eta)
    p_mat[3, 2] = p_mat[0, 1]
    p_mat[3, 3] = p_mat[1, 1]
    return p_mat


def _haskell(omega, rayp, nl, ipha, alpha, beta, rho, thick):
    e_inv = _e_inverse(omega, rho[-1], alpha[-1], beta[-1], rayp)
    p_mat = _propagator_sol(omega, rho[0], alpha[0], beta[0], rayp, thick[0])
    for i in range(1, nl):
        p_mat = _propagator_sol(
            omega, rho[i], alpha[i], beta[i], rayp, thick[i]) @ p_mat

    sl = e_inv @ p_mat if nl > 1 else e_inv
    denom = sl[2, 0] * sl[3, 1] - sl[2, 1] * sl[3, 0]
    if ipha >= 0:
        ur = sl[3, 1] / denom
        uz = -sl[3, 0] / denom
    else:
        ur = -sl[2, 1] / denom
        uz = sl[2, 0] / denom
    return ur, uz


def _fwd_seis(rayp, dt, npts, ipha, alpha, beta, rho, thick):
    npts = int(npts)
    nlay = thick.size
    ur_freq = np.zeros(npts, dtype=np.complex128)
    uz_freq = np.zeros(npts, dtype=np.complex128)
    nhalf = int(npts / 2 + 1)
    for i in range(1, nhalf):
        omega = 2 * np.pi * i / (npts * dt)
        ur_freq[i], uz_freq[i] = _haskell(
            omega, rayp, nlay, ipha, alpha, beta, rho, thick)
    return ur_freq, uz_freq


def _gaussian_taper(freq, gauss):
    return np.exp(- (np.pi * freq / gauss) ** 2)


def _waterlevel_deconvolution(radial, vertical, dt, gauss, water):
    radial_spec = np.fft.rfft(radial)
    vertical_spec = np.fft.rfft(vertical)
    freq = np.fft.rfftfreq(radial.size, d=dt)
    gfilter = _gaussian_taper(freq, gauss)

    denom = vertical_spec * np.conjugate(vertical_spec)
    level = water * np.max(np.abs(denom))
    denom = np.maximum(np.abs(denom), level)
    rf_spec = radial_spec * np.conjugate(vertical_spec) / denom
    rf_spec *= gfilter
    return np.fft.irfft(rf_spec, n=radial.size)


class RFminiModRF(object):
    """Forward modeling of receiver functions based on SeisPy (Joachim Saul).
    """
    def __init__(self, obsx, ref):
        self.ref = ref
        self.obsx = obsx
        self._init_obsparams()

        if self.ref in ['prf', 'seis']:
            self.modelparams = {'wtype': 'P'}
        elif self.ref in ['srf']:
            self.modelparams = {'wtype': 'SV'}

        self.modelparams.update(
            {'gauss': 1.0,
             'p': 6.4,
             'water': 0.001,
             'nsv': None
             })

        self.keys = {'z': '%.2f',
                     'vp': '%.4f',
                     'vs': '%.4f',
                     'rho': '%.4f',
                     'qp': '%.1f',
                     'qs': '%.1f',
                     'n': '%d'}

    def _init_obsparams(self):
        """Extract parameters from observed x-data (time vector).

        fsamp = sampling frequency in Hz
        tshft = time shift by which the RF is shifted to the left
        nsamp = number of samples, must be 2**x
        """

        # get fsamp
        deltas = np.round((self.obsx[1:] - self.obsx[:-1]), 4)
        if np.unique(deltas).size == 1:
            dt = float(deltas[0])
            self.fsamp = 1. / dt
        else:
            raise ValueError("Target: %s. Sampling rate must be constant."
                             % self.ref)
        # get tshft
        self.tshft = -self.obsx[0]

        # get nsamp
        ndata = self.obsx.size
        self.nsamp = int(2**int(np.ceil(np.log2(ndata * 2))))

    def write_startmodel(self, h, vp, vs, rho, modfile, **params):
        qp = params.get('qp', np.ones(h.size) * 500)
        qs = params.get('qs', np.ones(h.size) * 225)

        z = np.cumsum(h)
        z = np.concatenate(([0], z[:-1]))

        mparams = {'z': z, 'vp': vp, 'vs': vs, 'rho': rho,
                   'qp': qp, 'qs': qs}
        mparams = dict((a, b) for (a, b) in mparams.items()
                       if b is not None)
        pars = list(mparams.keys())

        nkey = 0
        header = []
        mline = []
        data = np.empty((len(pars), mparams[pars[0]].size))
        for key in ['z', 'vp', 'vs', 'rho', 'qp', 'qs']:
            if key in pars:
                header.append(key)
                mline.append(self.keys[key])
                data[nkey, :] = mparams[key]
                nkey += 1

        header = '\t'.join(header) + '\n'
        mline = '\t'.join(mline) + '\n'

        with open(modfile, 'w') as f:
            f.write(header)
            for i in np.arange(len(data[0])):
                f.write(mline % tuple(data.T[i]))

    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def compute_rf(self, h, vp, vs, rho, **params):
        """
        Compute RF using self.modelsparams (dict) for parameters.
        e.g. usage: self.set_modelparams(gauss=1.0)

        Parameters are:
        # z  depths of the top of each layer
        gauss: Gauss parameter
        water: water level
        p: angular slowness in sec/deg
        wtype: type of incident wave; must be 'P' or 'SV'
        nsv: tuple with near-surface S velocity and Poisson's ratio
            (will be computed by input model, if None)
        """
        gauss = self.modelparams['gauss']
        water = self.modelparams['water']
        p = self.modelparams['p']
        wtype = self.modelparams['wtype']
        nsv = self.modelparams['nsv']

        time = np.arange(self.nsamp) / self.fsamp - self.tshft
        rayp_s_per_km = float(p) * SEC_PER_DEG_TO_SEC_PER_KM
        ipha = 1 if wtype == 'P' else -1
        dt = 1. / self.fsamp
        ur_freq, uz_freq = _fwd_seis(
            rayp_s_per_km, dt, self.nsamp, ipha, vp, vs, rho, h)

        ur = np.fft.ifft(ur_freq).real[::-1] / self.nsamp
        uz = -np.fft.ifft(uz_freq).real[::-1] / self.nsamp
        qrfdata = _waterlevel_deconvolution(ur, uz, dt, gauss, water).astype(float)

        if nsv is not None:
            qrfdata *= float(vs[0]) / float(nsv)

        return time[:self.obsx.size], qrfdata[:self.obsx.size]

    def run_model(self, h, vp, vs, rho, **params):

        assert h.size == vp.size == vs.size == rho.size

        h = h.astype(float)
        vp = vp.astype(float)
        vs = vs.astype(float)
        rho = rho.astype(float)

        time, qrf = self.compute_rf(h, vp, vs, rho, **params)
        return time, qrf
