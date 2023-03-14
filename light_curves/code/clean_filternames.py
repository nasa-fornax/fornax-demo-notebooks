def clean_filternames(search_result, numlc):
    """Simplify mission name from a combined list
    
    Mission names are returned including quarter numbers, remove those to
    simplify. For this use case, we really only need to know which mission, 
    not which quarter.
    
    Parameters
    ----------
    search_result : lightkurve object
        object detailing the light curve object found with lightkurve
    numlc : int
        index of the light curve we are working on
        
    Returns
    -------
    filtername : str
        name of the mission without quarter information
    """            
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