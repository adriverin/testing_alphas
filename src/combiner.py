import pandas as pd

def combine_alphas(alpha_calculator, alpha_list, method='mean'):
    """
    Combines multiple alpha signals into a single "mega-alpha" signal.
    This version uses a robust for-loop for index alignment to be completely safe.
    """
    print(f"\n--- Combining {len(alpha_list)} Alphas ---")
    
    all_alpha_signals = []
    
    # 1. Calculate each individual alpha signal
    for alpha_name in alpha_list:
        if hasattr(alpha_calculator, alpha_name):
            print(f"  Calculating {alpha_name}...")
            try:
                signal = getattr(alpha_calculator, alpha_name)().dropna()
                if not signal.empty:
                    signal.name = alpha_name
                    all_alpha_signals.append(signal)
                else:
                    print(f"    -> WARNING: {alpha_name} produced no valid signals (all NaN).")
            except Exception as e:
                print(f"    -> FAILED to calculate {alpha_name}: {e}")
        else:
            print(f"  -> WARNING: {alpha_name} not found in calculator.")
            

    # 2. Check if we have enough signals to proceed
    if len(all_alpha_signals) < 2:
        print("\nERROR: Need at least two successful alpha signals to combine. Exiting combination.")
        return pd.Series(dtype=float)
        
    # 3. Find the common index using a simple and robust for-loop
    # Start with the index of the first successful signal
    common_index = all_alpha_signals[0].index
    
    # Iteratively find the intersection with all other signal indices
    for i in range(1, len(all_alpha_signals)):
        common_index = common_index.intersection(all_alpha_signals[i].index)
        
    if common_index.empty:
        print("No common data points found across all selected alphas. Cannot combine.")
        return pd.Series(dtype=float)

    # 4. Align all signals to this common index before creating the DataFrame
    aligned_signals = {s.name: s.reindex(common_index) for s in all_alpha_signals}
    alpha_df = pd.DataFrame(aligned_signals)
    
    # 5. Rank each alpha signal column cross-sectionally
    ranked_alpha_df = alpha_df.groupby(level='date').rank(pct=True)
    
    # 6. Average the ranks to get the final signal
    if method == 'mean':
        mega_alpha_signal = ranked_alpha_df.mean(axis=1)
    else:
        raise NotImplementedError(f"Combination method '{method}' is not implemented.")
        
    print("--- Alpha Combination Complete ---")
    
    return mega_alpha_signal.dropna()