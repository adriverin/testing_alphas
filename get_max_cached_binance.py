from multi_crypto_ml_training import create_maximum_cache_for_assets

# This will:
# 1. Auto-detect actual date ranges for each asset
# 2. Download maximum available data 
# 3. Create detailed information files
# 4. Handle assets that don't exist for early dates


create_maximum_cache_for_assets(    
    assets=[
        'BTC-USD',  # Bitcoin
        'ETH-USD',  # Ethereum
        'SOL-USD',  # Solana
        'ADA-USD',  # Cardano
        'AVAX-USD',  # Avalanche
        'BNB-USD',  # Binance Coin
        'XRP-USD',  # Ripple
        'LTC-USD',  # Litecoin
        'LINK-USD',  # Chainlink
        'XLM-USD',  # Stellar
        'ATOM-USD',  # Cosmos
        'HBAR-USD',  # Hedera
        'BCH-USD',  # Bitcoin Cash
        'DOT-USD',  # Polkadot
        'UNI-USD',  # Uniswap
        'AAVE-USD',  # Aave
        'SCRT-USD',  # Secret
        'ALGO-USD',  # Algorand
        'VET-USD',  # VeChain
        'XTZ-USD',  # Tezos
        #meme coins
        'DOGE-USD',  # Dogecoin
        'PEPE-USD',  # Pepe
        'SHIB-USD',   # Shiba Inu
        'BONK-USD',  # Bonk
        'WIF-USD',  # dogwifhat
        'FLOKI-USD',  # Floki
    ],
    
    interval="1h",
    start="2010-01-01",    # Request way back (auto-adjusts to actual)
    end="2030-12-31",      # Request far future (auto-adjusts to actual)
    create_info_file=True  # Creates the info files
)