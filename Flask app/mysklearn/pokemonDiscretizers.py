import numpy as np

def attack_discretizer(x):
    """Runs a list of attack values through a discretizer
       [5.0, 40.2, 75.4, 110.6, 145.8, 181.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 145.8:
            classification.append(5)
        elif val > 110.6:
            classification.append(4)
        elif val > 75.4:
            classification.append(3)
        elif val > 40.2:

            classification.append(2)
        else:
            classification.append(1)
    return classification

def base_happiness_discretizer(x):
    """Runs a list of base_happiness values through a discretizer
       [0.0, 46.67, 93.33, 140.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 93.33:
            classification.append(3)
        elif val > 46.67:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def base_total_discretizer(x):
    """Runs a list of bast_total values through a discretizer
       [180.0, 288.0, 396.0, 504.0, 612.0, 720.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 612:
            classification.append(5)
        elif val > 504:
            classification.append(4)
        elif val > 396:
            classification.append(3)
        elif val > 288:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def capture_rate_discretizer(x):
    """Runs a list of capture_rate values through a discretizer
       [3.0, 53.4, 103.8, 154.2, 204.6, 255.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 204.6:
            classification.append(5)
        elif val > 154.2:
            classification.append(4)
        elif val > 103.8:
            classification.append(3)
        elif val > 53.4:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def defense_discretizer(x):
    """Runs a list of defense values through a discretizer
       [5.0, 50.0, 95.0, 140.0, 185.0, 230.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 185:
            classification.append(5)
        elif val > 140:
            classification.append(4)
        elif val > 95:
            classification.append(3)
        elif val > 50:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def experience_growth_discretizer(x):
    """Runs a list of experience_growth values through a discretizer
       [600000.0, 808000.0, 1016000.0, 1224000.0, 1432000.0, 1640000.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 1432000:
            classification.append(5)
        elif val > 1224000:
            classification.append(4)
        elif val > 1016000:
            classification.append(3)
        elif val > 808000:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def height_m_discretizer(x):
    """Runs a list of height_m values through a discretizer
       [0.1, 2.09, 4.08, 6.07, 8.06, 10.05, 12.04, 14.03, 16.02, 18.01, 20.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 18.01:
            classification.append(10)
        elif val > 16.02:
            classification.append(9)
        elif val > 14.03:
            classification.append(8)
        elif val > 12.04:
            classification.append(7)
        elif val > 10.05:
            classification.append(6)
        elif val > 8.06:
            classification.append(5)
        elif val > 6.07:
            classification.append(4)
        elif val > 4.08:
            classification.append(3)
        elif val > 2.09:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def hp_discretizer(x):
    """Runs a list of hp values through a discretizer
       [1.0, 51.8, 102.6, 153.4, 204.2, 255.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 204.2:
            classification.append(5)
        elif val > 153.4:
            classification.append(4)
        elif val > 102.6:
            classification.append(3)
        elif val > 51.8:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def sp_attack_discretizer(x):
    """Runs a list of sp_attack values through a discretizer
       [10.0, 42.6, 75.2, 107.8, 140.4, 173.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 140.4:
            classification.append(5)
        elif val > 107.8:
            classification.append(4)
        elif val > 75.2:
            classification.append(3)
        elif val > 42.6:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def sp_defense_discretizer(x):
    """Runs a list of sp_defense values through a discretizer
       [20.0, 62.0, 104.0, 146.0, 188.0, 230.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 188:
            classification.append(5)
        elif val > 146:
            classification.append(4)
        elif val > 104:
            classification.append(3)
        elif val > 62:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def speed_discretizer(x):
    """Runs a list of speed values through a discretizer
       [5.0, 44.0, 83.0, 122.0, 161.0, 200.0]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 161:
            classification.append(5)
        elif val > 122:
            classification.append(4)
        elif val > 83:
            classification.append(3)
        elif val > 44:
            classification.append(2)
        else:
            classification.append(1)
    return classification

def weight_kg_discretizer(x):
    """Runs a list of weight_kg values through a discretizer
        [0.1, 100.08, 200.06, 300.04, 400.02, 500.0, 599.98, 699.96, 799.94, 899.92, 999.9]
    Args:
        x (list): the list of values to discretize
    Returns:
        classification: The list of discretizied values
    """
    classification = []
    for val in x:
        if val > 899.92:
            classification.append(10)
        elif val > 799.94:
            classification.append(9)
        elif val > 699.96:
            classification.append(8)
        elif val > 599.98:
            classification.append(7)
        elif val > 500.0:
            classification.append(6)
        elif val > 400.02:
            classification.append(5)
        elif val > 300.04:
            classification.append(4)
        elif val > 200.06:
            classification.append(3)
        elif val > 100.08:
            classification.append(2)
        else:
            classification.append(1)
    return classification

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