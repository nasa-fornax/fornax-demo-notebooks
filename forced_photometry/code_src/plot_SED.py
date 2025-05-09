import matplotlib.pyplot as plt


#function to display an SED 
def plot_SED(obj, df):

    #make super simple plot
    wavelength = [3.6, 4.5, 5.8, 8.0]
    #FUV ~1500Angstroms, NUV ~2300Angstroms or 0.15 & 0.23 microns
    flux = [df.ch1flux[obj],df.ch2flux[obj],df.ch3flux[obj],df.ch4flux[obj]]
    fluxerr = [df.ch1flux_unc[obj],df.ch2flux_unc[obj],df.ch3flux_unc[obj],df.ch4flux_unc[obj]]

    #fudge the uncertainties higher until I get the uncertainty function working
    fluxerr = [i * 5 for i in fluxerr]

    #plot as log wavenlength vs. log flux to eventually include Galex
    fig, ax = plt.subplots()
    ax.set_xscale("log", nonpositive='clip')
    ax.set_yscale("log", nonpositive='clip')
    ax.errorbar(wavelength, flux, yerr = fluxerr)

    #set up labels
    ax.set(xlabel = 'Wavelength (microns)', ylabel = r"Flux ($\mu$Jy)", title = 'SED')
    plt.show()
    
    return


