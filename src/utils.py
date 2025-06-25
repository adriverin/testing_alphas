import pandas as pd

def generate_date_intervals(start_date, end_date, n):
    """
    Generates n intervals of start and end dates between the given start and end date.
    
    Parameters:
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    n (int): The number of intervals to generate.
    
    Returns:
    list of tuples: A list containing n tuples of (start, end) date intervals.
    """
    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    total_days = (end_date_dt - start_date_dt).days
    interval_length = total_days // n
    
    intervals = []
    for i in range(n):
        interval_start = start_date_dt + pd.Timedelta(days=i * interval_length)
        interval_end = start_date_dt + pd.Timedelta(days=(i + 1) * interval_length) if i < n - 1 else end_date_dt
        intervals.append((interval_start.strftime('%Y-%m-%d'), interval_end.strftime('%Y-%m-%d')))
    
    return intervals