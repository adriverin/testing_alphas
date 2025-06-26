from src.get_stock_data import get_stock_data
from src.alpha101 import Alpha101
from src.analysis import generate_full_report, generate_interval_report, generate_summary_html_report
from src.utils import generate_date_intervals



if __name__ == "__main__":


    # def run_walk_forward(start, end, n_splits=5):
    #     print(" ")
    #     print(" ")
    #     print("STARTING")
    #     print("1. Getting real stock data...")

    #     price_data = get_stock_data(sp100_tickers, start_date=start, end_date=end, cache_path='stock_data.parquet')

    #     for start, end in generate_date_intervals(start, end, n_splits):
    #         # price_data[]

    #     for i in range(1, 103):
    #         alpha_name = f'alpha{i:03d}'
    #         if hasattr(alpha_calculator, alpha_name) and callable(getattr(alpha_calculator, alpha_name)):
    #             print(f"Processing {alpha_name}...")
    #             alpha_series = getattr(alpha_calculator, alpha_name)().dropna()
                
    #             if not alpha_series.empty:
    #                 generate_full_report(alpha_calculator, price_data, pdf_path=f'test_reports/{alpha_name}_performance_report.pdf')  

    #     for start, end in generate_date_intervals(start, end, n_splits):

    #         if not price_data.empty:
    #             print(" ")
    #             print("2. Initializing the Alpha101 calculator...")
    #             alpha_calculator = Alpha101(price_data) 
                
    #             for i in range(1, 103):
    #                 alpha_name = f'alpha{i:03d}'
    #                 if hasattr(alpha_calculator, alpha_name) and callable(getattr(alpha_calculator, alpha_name)):
    #                     print(f"Processing {alpha_name}...")
    #                     alpha_series = getattr(alpha_calculator, alpha_name)().dropna()
                        
    #                     if not alpha_series.empty:
    #                         generate_full_report(alpha_calculator, price_data, pdf_path=f'test_reports/{alpha_name}_performance_report.pdf')   

                                

    sp100_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'PG',
        'UNH', 'HD', 'MA', 'BAC', 'PFE', 'XOM', 'CVX', 'KO', 'PEP', 'WMT',
        'META'
    ]

    # sp100_tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'DOGE-USD', 'SOL-USD', 'DOT-USD', 'SHIB-USD', 'ADA-USD', 'LTC-USD', 'BNB-USD', 'AVAX-USD', 'PEPE24478-USD']
    # sp100_tickers = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'LTC-USD', 'BNB-USD']

    # sp100_tickers = [
    # "AAPL","ABBV","ABT","ACN","ADBE","AIG","AMD","AMGN","AMT","AMZN",
    # "AVGO","AXP","BA","BAC","BK","BKNG","BLK","BMY","C",
    # "CAT","CHTR","CL","CMCSA","COF","COP","COST","CRM","CSCO","CVS",
    # "CVX","DE","DHR","DIS","DUK","EMR","FDX","GD","GE","GILD",
    # "GM","GOOG","GOOGL","GS","HD","HON","IBM","INTC","INTU",
    # "JNJ","JPM","KO","LIN","LLY","LMT","LOW","MA","MCD","MDLZ",
    # "MDT","MET","META","MMM","MO","MRK","MS","MSFT","NFLX",
    # "NKE","NOW","NVDA","ORCL","PEP","PFE","PG","PLTR","PM",
    # "QCOM","RTX","SBUX","SCHW","SO","SPG","T","TGT","TMO","TMUS",
    # "TSLA","TXN","UNH","UNP","UPS","USB","V","VZ","WFC","WMT","XOM"
    # ]
    start = '2011-01-01'
    end = '2025-01-01'

    print(" ")
    print(" ")
    print("STARTING")
    print("1. Getting real stock data...")
    price_data = get_stock_data(sp100_tickers, start_date=start, end_date=end, cache_path='stock_data.parquet')

    # if not price_data.empty:
    #     print("\n2. Initializing the Alpha101 calculator...")
    #     alpha_calculator = Alpha101(price_data)
        
    #     generate_full_report(alpha_calculator, price_data, pdf_path=f'test_reports/alpha_performance_report_{start}_{end}.pdf')

    # a = generate_date_intervals(start, end, 10)
    # print(a)

    # if not price_data.empty:
    #     for start, end in generate_date_intervals(start, end, 5):
    #         print(f"Running for {start} to {end}")
    #         price_data = get_stock_data(sp100_tickers, start_date=start, end_date=end, cache_path='stock_data.parquet')
    #         if not price_data.empty:
    #             print("\n2. Initializing the Alpha101 calculator...")
    #             alpha_calculator = Alpha101(price_data)
    #             generate_full_report(alpha_calculator, price_data, pdf_path=f'test_reports/alpha_performance_report_{start}_{end}.pdf')




    if not price_data.empty:
        print("\n2. Initializing the Alpha101 calculator...")
        alpha_calculator = Alpha101(price_data)
        
        # Define the intervals you want to test
        # For example, let's split the whole period into 4 chunks
        number_of_intervals = 20
        intervals_to_test = generate_date_intervals(start, end, number_of_intervals)
        
        print(f"\nGenerated {len(intervals_to_test)} testing intervals:")
        for s, e in intervals_to_test:
            print(f"  - {s} to {e}")
            
        # Run the new interval-based report generator
        firstAlpha = 200
        lastAlpha = 300

        generate_interval_report(alpha_calculator, price_data, intervals_to_test, first_alpha=firstAlpha, last_alpha=lastAlpha+1)
        generate_summary_html_report(alpha_calculator, price_data, intervals_to_test, first_alpha=firstAlpha, last_alpha=lastAlpha+1)
        generate_full_report(alpha_calculator, price_data, first_alpha=firstAlpha, last_alpha=lastAlpha+1)
