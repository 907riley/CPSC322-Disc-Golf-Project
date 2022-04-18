def attack_discretizer():

    pass

def base_happiness_discretizer():

    pass

def base_total_discretizer():

    pass

def capture_rate_discretizer():

    pass

def defense_discretizer():

    pass

def experience_growth_discretizer():

    pass

def height_m_discretizer():

    pass

def hp_discretizer():

    pass

def sp_attack_discretizer():

    pass

def sp_defense_discretizer():

    pass

def speed_discretizer():

    pass

def weight_kg_discretizer():

    pass

def compute_bin_frequencies(values, cutoffs):
    """Computes the bin frequencies for a list of values

    Source:
        Taken from our in class notes on histograms
    
    Args:
        values (list): list of values compute on
        cutoffs (list): list of cutoffs to use
    
    Returns: 
        freqs: list of frequencies
    """
    freqs = [0 for _ in range(len(cutoffs) - 1)] # because we have N + 1 cutoffs

    for value in values:
        if value == max(values):
            freqs[-1] += 1 # add one to the last bin count
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1 
                    # add one to this bin defined by [cutoffs[i], cutoffs[i+1]]
    return freqs

def compute_equal_width_cutoffs(values, num_bins):
    """Computes the equal width cutoffs for a list of values and 
        number of bins

    Source:
        Taken from our in class notes on histograms
    
    Args:
        values (list): list of values compute on
        num_bins (int): number of bins to use
    
    Returns: 
        cutoff: list of cutoffs for the bins
    """

    values_range = max(values) - min(values)
    bin_width = values_range / num_bins # float
    # since bin_width is a float, we shouldn't use range() to generate a list
    # of cutoffs, use np.arange()
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values)) # exactly the max(values)
    # to handle round off error...
    # if your application allows, we should convert to int
    # or optionally round them
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs