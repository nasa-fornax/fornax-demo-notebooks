from tractor import Flux, PixPos, PointSource


# function to determine what type of source it is from catalog
def determine_source_type(ra, dec, df_type, fid_flux, x1, y1):
    # make all sources point sources for now
    # use fiducial flux as first guess of source flux in different bands

    src = PointSource(PixPos(x1, y1), Flux(fid_flux))
    return src
