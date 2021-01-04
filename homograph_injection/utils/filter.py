from timeit import default_timer as timer

def build_filter_dict(filter, min_str_length, max_str_length, min_cardinality, max_cardinality, remove_numerical_vals):
    if filter:
        filter_dict = {}
        if min_str_length != None:
            filter_dict['min_str_length'] = min_str_length
        if max_str_length != None:
            filter_dict['max_str_length'] = max_str_length
        if min_cardinality != None:
            filter_dict['min_cardinality'] = min_cardinality
        if max_cardinality != None:
            filter_dict['max_cardinality'] = max_cardinality
        if remove_numerical_vals != None:
            filter_dict['remove_numerical_vals'] = remove_numerical_vals
        return filter_dict
    else:
        return None

def is_number_tryexcept(s):
    """ 
    Returns True if string `s` is a number.

    Taken from: https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def filter_values(value_stats_dict, filter_dict):
    '''
    Given the value_stats_dict updated it by filtering out
    keys given a filter_dict specifying the filter rules.
    '''
    values_for_removal = set()
    numerical_values_removed = 0
    short_string_values_removed = 0
    long_sting_values_removed = 0
    small_cardinality_values_removed = 0
    large_cardinality_values_removed = 0

    print('Filtering values...')
    start = timer()
    for val in value_stats_dict:
        if 'min_cardinality' in filter_dict:
            if value_stats_dict[val]['cardinality'] < filter_dict['min_cardinality']:
                values_for_removal.add(val)
                small_cardinality_values_removed += 1
        if 'max_cardinality' in filter_dict:
            if value_stats_dict[val]['cardinality'] > filter_dict['max_cardinality']:
                values_for_removal.add(val)
                large_cardinality_values_removed += 1

        if is_number_tryexcept(val):
            if 'remove_numerical_vals' in filter_dict:
                values_for_removal.add(val)
                numerical_values_removed += 1
        else:
            if 'min_str_length' in filter_dict and len(val) < filter_dict['min_str_length']:
                values_for_removal.add(val)
                short_string_values_removed += 1
            elif 'max_str_length' in filter_dict and len(val) > filter_dict['max_str_length']:
                values_for_removal.add(val)
                long_sting_values_removed += 1
    print('Finished filtering values. \nElapsed time:', timer()-start, 'seconds')

    # Remove the filtered values from the value_stats_dict
    [value_stats_dict.pop(key) for key in values_for_removal] 

    print('Removed', len(values_for_removal), 'values in total.')
    print('Removed', numerical_values_removed, 'values with numerical values',
        short_string_values_removed, 'values with short strings', long_sting_values_removed, 'values with long strings',
        small_cardinality_values_removed, 'values with small cardinality', large_cardinality_values_removed, 'values with large cardinality')
    print(len(value_stats_dict), 'values are available for selection')
    return value_stats_dict