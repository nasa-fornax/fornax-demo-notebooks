import astropy.io.fits as fits


#nway-write-header.py adapted from github

def nway_write_header(catalog_fits, cat_name, skyarea):
    f = fits.open(catalog_fits)
    print('current', f[1].name, 'SKYAREA:', f[1].header.get('SKYAREA', None))
    f[1].name = cat_name
    f[1].header['SKYAREA'] = float(skyarea)
    print('new    ', f[1].name, 'SKYAREA:', f[1].header.get('SKYAREA', None))

    f.writeto(catalog_fits, overwrite = "True")
    
