import pandas as pd
from pathlib import Path

def process_gage_data(rain_dfs: list[pd.DataFrame], stream_df: pd.DataFrame, smooth: bool = False, roll_n: int = 3):
    """
    Process rain-gage and stream-gage dataframes to sync timestamps and optionally smooth data.

    Args:
        rain_dfs (list[pd.DataFrame]): List of rain-gage DataFrames.
        stream_df (pd.DataFrame): Stream-gage DataFrame.
        smooth (bool): Whether to apply rolling smoothing.
        roll_n (int): Window size for rolling smoothing.

    Returns:
        tuple: Synced (and optionally smoothed) stream DataFrame and list of rain DataFrames.
    """
    # Ensure stream data is sorted
    stream_df['collect_date'] = pd.to_datetime(stream_df['collect_date'])
    stream_df = stream_df[['collect_date', 'stage']].sort_values('collect_date').reset_index(drop=True)

    # Ensure rain data is sorted
    for i, df in enumerate(rain_dfs):
        rain_dfs[i]['collect_date'] = pd.to_datetime(df['collect_date'])
        rain_dfs[i] = df[['collect_date', 'rain_amount']].sort_values('collect_date').reset_index(drop=True)

    # Find overlapping time range among all datasets
    min_time = stream_df['collect_date'].min()
    max_time = stream_df['collect_date'].max()
    for df in rain_dfs:
        min_time = max(min_time, df['collect_date'].min())
        max_time = min(max_time, df['collect_date'].max())

    # Filter each dataframe to the overlapping range
    stream_df = stream_df[(stream_df['collect_date'] >= min_time) & (stream_df['collect_date'] <= max_time)].reset_index(drop=True)
    for i, df in enumerate(rain_dfs):
        rain_dfs[i] = df[(df['collect_date'] >= min_time) & (df['collect_date'] <= max_time)].reset_index(drop=True)

    # Create combined set of timestamps from all filtered dataframes
    all_times = pd.Series(stream_df['collect_date'].tolist())
    for df in rain_dfs:
        all_times = pd.concat([all_times, pd.Series(df['collect_date'].tolist())], ignore_index=True)
    all_times = all_times.drop_duplicates().sort_values().reset_index(drop=True)

    # Matching helper function
    def match_to_timeline(df, timeline, value_col, tolerance_minutes=30):
        tol = pd.Timedelta(minutes=tolerance_minutes)
        temp = pd.DataFrame({'collect_date': timeline})
        merged = pd.merge_asof(temp, df.sort_values('collect_date').reset_index(drop=True),
                               on='collect_date', direction='nearest', tolerance=tol)
        return merged[value_col]

    # Attempt matching across all datasets using escalating tolerances
    for tol in (30, 60, 120):
        timeline = all_times.copy()
        stream_matched = match_to_timeline(stream_df, timeline, 'stage', tolerance_minutes=tol)
        rain_matched_dict = {}
        for i, df in enumerate(rain_dfs):
            rain_matched_dict[i] = match_to_timeline(df, timeline, 'rain_amount', tolerance_minutes=tol)

        valid_mask = ~stream_matched.isna()
        for series in rain_matched_dict.values():
            valid_mask &= ~series.isna()

        final_timeline = timeline[valid_mask].reset_index(drop=True)
        if len(final_timeline) > 0:
            stream_matched = stream_matched[valid_mask].reset_index(drop=True)
            for i in rain_matched_dict:
                rain_matched_dict[i] = rain_matched_dict[i][valid_mask].reset_index(drop=True)
            chosen_tol = tol
            break
    else:
        raise RuntimeError("No common timestamps found within 120 minute tolerance")

    # Optionally smooth series
    if smooth:
        stream_matched = stream_matched.rolling(roll_n, min_periods=1).mean()
        for i in rain_matched_dict:
            rain_matched_dict[i] = rain_matched_dict[i].rolling(roll_n, min_periods=1).mean()

    # Prepare output DataFrames
    stream_out_df = pd.DataFrame({'collect_date': final_timeline, 'stage': stream_matched})
    rain_out_dfs = []
    for i, series in rain_matched_dict.items():
        rain_out_dfs.append(pd.DataFrame({'collect_date': final_timeline, 'rain_amount': series}))

    return stream_out_df, rain_out_dfs

