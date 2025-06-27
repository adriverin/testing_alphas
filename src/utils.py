import pandas as pd
import pandas_datareader.data as web # For Fama-French data

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





def get_fama_french_factors(start_date, end_date):
    """
    Downloads Fama-French 3-factor daily data.
    """
    print("\n--- Downloading Fama-French 3-Factor Data ---")
    try:
        # The 'F-F_Research_Data_Factors_daily' dataset gives Mkt-RF, SMB, HML, and RF
        ff_data = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start=start_date, end=end_date)
        # The library returns a dictionary of DataFrames, we want the first one (daily data)
        ff_df = ff_data[0]
        # The values are in percentages, so divide by 100
        ff_df = ff_df / 100
        # ff_df.index = ff_df.index.to_timestamp() # Convert the PeriodIndex to a DatetimeIndex
        print("Fama-French data downloaded successfully.")
        return ff_df
    except Exception as e:
        print(f"Could not download Fama-French data. Error: {e}")
        print("Please ensure you have the 'lxml' package installed (`pip install lxml`).")
        return None