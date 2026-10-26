# calculate an uncertainty as a combination of
# tractor variance as reported by tractor
# background noise
# poisson noise

# not currently complete, it is just the start of where I think this could go to be more rigorous

def calc_irac_uncertainty(ch, flux, skynoise, tractor_std):
    # try putting everything in units of electrons to calculate the noise

    # clear up some variables first
    # cryo gain for 4 channels
    # think cosmos2015 is all cryo; really only matters for ch1
    cryo_gain = [3.3, 3.7, 3.8, 3.8]  # electrons/DN

    # mosaic pixel scale
    pix_scale = 0.6  # arcseconds
    exptime = 3.8 * 60 * 60  # average exposure time in (s) over all IRAC data =3.8hrs from Laigle et al. 2016

    # fluxconv #from the headers or from the IRAC data handbook Table 4.2
    fluxconv = [0.1069, 0.1382, 0.5858, 0.2026]  # (MJy/sr) / (DN/s)

    # measurement area in pixels
    rad_prf = 3  # radius is 3 pixels, this is all a bit wishywashy
    A = pi * (rad_prf**2)

    # -------------------
    # background noise
    # skynoise comes in with units of MJy/sr
    # need to work the area in here***
    bkg_electrons = sknoise * cryo_gain[ch] * exptime / flux_conv[ch]

    # ------------------
    # poisson noise
    # flux comes in units of MJy/sr
    electrons = flux * cryo_gain[ch] * exptime / flux_conv[ch]
    poisson_noise = np.sqrt(electrons)

    # ------------------
    # tractor uncertainty
    # comes in units of ???

    # ------------------
    # add the uncertainties in quadrature
    unc = numpy.sqrt(bkg_noise**2 + poisson_noise**2 + tractor_std**2)

    return unc
