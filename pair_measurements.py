import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# smoothing parameters
SMOOTH = True          # whether to produce smoothed csvs
ROLL_N = 3              # number of timestamps to include in rolling average

# data directory (adjust if necessary or pass as arg)
data_dir = Path(__file__).parent / 'data'

# find the stream file (should be one)
stream_files = list(data_dir.glob('stream-gages_*.csv'))
if not stream_files:
    raise RuntimeError(f"No stream-gages CSV found in {data_dir}")
if len(stream_files) > 1:
    print("Warning: multiple stream files found, using the first one:", stream_files[0])
stream_gages_file = stream_files[0]

# find rain files
rain_files = list(data_dir.glob('rain-gages_*.csv'))
if not rain_files:
    raise RuntimeError(f"No rain-gages CSVs found in {data_dir}")

# load stream data
print("Loading stream-gages file...", stream_gages_file)
stream_df = pd.read_csv(stream_gages_file)
stream_df['Date'] = pd.to_datetime(stream_df['Date'])
stream_df = stream_df[['Date', 'Stage']].sort_values('Date').reset_index(drop=True)

# load all rain dataframes into a dict keyed by filename
rain_dfs = {}
for rf in rain_files:
    print("Loading rain-gages file...", rf)
    df = pd.read_csv(rf)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Rain Amount (in)']].sort_values('Date').reset_index(drop=True)
    rain_dfs[rf] = df

print(f"Stream-gages: {len(stream_df)} records from {stream_df['Date'].min()} to {stream_df['Date'].max()}")
for rf, df in rain_dfs.items():
    print(f"Rain-gages ({rf.name}): {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")

# Find overlapping time range among all datasets
min_time = stream_df['Date'].min()
max_time = stream_df['Date'].max()
for df in rain_dfs.values():
    min_time = max(min_time, df['Date'].min())
    max_time = min(max_time, df['Date'].max())
print(f"Overlapping time range across all datasets: {min_time} to {max_time}")

# Filter each dataframe to that range
stream_df = stream_df[(stream_df['Date'] >= min_time) & (stream_df['Date'] <= max_time)].reset_index(drop=True)
for key in list(rain_dfs.keys()):
    df = rain_dfs[key]
    rain_dfs[key] = df[(df['Date'] >= min_time) & (df['Date'] <= max_time)].reset_index(drop=True)

print("After filtering to overlapping range:")
print(f"  Stream-gages: {len(stream_df)} records")
for rf, df in rain_dfs.items():
    print(f"  {rf.name}: {len(df)} records")

# create combined set of timestamps from all filtered dataframes
all_times = pd.Series(stream_df['Date'].tolist())
for df in rain_dfs.values():
    all_times = pd.concat([all_times, pd.Series(df['Date'].tolist())], ignore_index=True)
all_times = all_times.drop_duplicates().sort_values().reset_index(drop=True)
print(f"Total unique timestamps (union): {len(all_times)}")

# matching helper that attaches nearest values from a df to a timeline

def match_to_timeline(df, timeline, value_col, tolerance_minutes=30):
    """
    Given a dataframe with 'Date' and a value column, return a Series aligned to
    the provided timeline (pd.Series of timestamps). Nearest neighbor matching
    within tolerance; returns NaN for no match.
    """
    tol = pd.Timedelta(minutes=tolerance_minutes)
    temp = pd.DataFrame({'Date': timeline})
    merged = pd.merge_asof(temp, df.sort_values('Date').reset_index(drop=True),
                           on='Date', direction='nearest', tolerance=tol)
    return merged[value_col]

# attempt matching across all datasets using escalating tolerances
for tol in (30, 60, 120):
    print(f"\nTrying tolerance = {tol} minutes...")
    # build timeline using union of all dates
    timeline = all_times.copy()
    # compute matched values for stream and each rain file
    stream_matched = match_to_timeline(stream_df, timeline, 'Stage', tolerance_minutes=tol)
    rain_matched_dict = {}
    for rf, df in rain_dfs.items():
        rain_matched_dict[rf] = match_to_timeline(df, timeline, 'Rain Amount (in)', tolerance_minutes=tol)
    # identify timestamps where all series have non-null values
    valid_mask = ~stream_matched.isna()
    for series in rain_matched_dict.values():
        valid_mask &= ~series.isna()
    final_timeline = timeline[valid_mask].reset_index(drop=True)
    print(f"  valid common timestamps: {len(final_timeline)}")
    if len(final_timeline) > 0:
        # store matched series for this tolerance and break
        stream_matched = stream_matched[valid_mask].reset_index(drop=True)
        for rf in rain_matched_dict:
            rain_matched_dict[rf] = rain_matched_dict[rf][valid_mask].reset_index(drop=True)
        chosen_tol = tol
        break
else:
    raise RuntimeError("No common timestamps found within 120 minute tolerance")

print(f"Using tolerance {chosen_tol} minutes with {len(final_timeline)} records")

# optionally smooth series
if SMOOTH:
    print(f"Applying rolling smoothing with window={ROLL_N}")
    stream_smoothed = stream_matched.rolling(ROLL_N, min_periods=1).mean()
    rain_smoothed_dict = {}
    for rf, series in rain_matched_dict.items():
        rain_smoothed_dict[rf] = series.rolling(ROLL_N, min_periods=1).mean()

# prepare output directory
output_dir = Path(__file__).parent / 'data' / 'paired'
output_dir.mkdir(exist_ok=True)

# write out stream output
stream_out_df = pd.DataFrame({'Date': final_timeline, 'Stage': stream_matched})
stream_out_path = output_dir / stream_gages_file.name.replace('.csv','_synced.csv')
stream_out_df.to_csv(stream_out_path, index=False)
print(f"Wrote stream output: {stream_out_path}")

# write out each rain output
for rf, series in rain_matched_dict.items():
    rain_out_df = pd.DataFrame({'Date': final_timeline, 'Rain Amount (in)': series})
    rain_out_path = output_dir / rf.name.replace('.csv','_synced.csv')
    rain_out_df.to_csv(rain_out_path, index=False)
    print(f"Wrote rain output: {rain_out_path}")

# if smoothing requested, write smoothed files
if SMOOTH:
    stream_smoothed_df = pd.DataFrame({'Date': final_timeline, 'Stage': stream_smoothed})
    stream_smoothed_path = output_dir / stream_gages_file.name.replace('.csv','_synced_smoothed.csv')
    stream_smoothed_df.to_csv(stream_smoothed_path, index=False)
    print(f"Wrote stream smoothed output: {stream_smoothed_path}")

    for rf, series in rain_smoothed_dict.items():
        rain_sm_df = pd.DataFrame({'Date': final_timeline, 'Rain Amount (in)': series})
        rain_sm_path = output_dir / rf.name.replace('.csv','_synced_smoothed.csv')
        rain_sm_df.to_csv(rain_sm_path, index=False)
        print(f"Wrote rain smoothed output: {rain_sm_path}")

# show samples
print("\nSample of synced data (first 5 rows):")
print(stream_out_df.head().to_string())

