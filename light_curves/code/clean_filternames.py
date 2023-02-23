def clean_filternames(search_result, numlc):
                
    filtername = str(search_result[numlc].mission)
    #clean this up a bit so all Kepler quarters etc., get the same filtername
    #we don't need to track the individual names for the quarters, just need to know which mission it is
    if 'Kepler' in filtername:
        filtername = 'Kepler'
    if 'TESS' in filtername:
        filtername = 'TESS'
    if 'K2' in filtername:
        filtername = 'K2'
    return(filtername)