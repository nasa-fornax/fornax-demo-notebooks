## Plotting stuff
def setup_text_plots():
    """
    This function adjusts matplotlib settings so that all figures have a uniform format and look.
    """
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.labelpad'] = 7
    mpl.rcParams['xtick.major.pad'] = 7
    mpl.rcParams['ytick.major.pad'] = 7
    mpl.rcParams['xtick.minor.visible'] = True
    mpl.rcParams['ytick.minor.visible'] = True
    mpl.rcParams['xtick.minor.top'] = True
    mpl.rcParams['xtick.minor.bottom'] = True
    mpl.rcParams['ytick.minor.left'] = True
    mpl.rcParams['ytick.minor.right'] = True
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['ytick.major.size'] = 5
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    #mpl.rc('text', usetex=True)
    mpl.rc('font', family='serif')
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['hatch.linewidth'] = 1

