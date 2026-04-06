# data_loader.py — load Milan CSV, create windows, split train/test

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import DATASET_PATH, HOURS_PER_DAY, TRAIN_DAYS, TOTAL_DAYS, W, L


def load_data(path=None):
    """Load Milan traffic CSV and return (values, base_date) tuple."""
    if path is None:
        path = DATASET_PATH

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Accept both 'traffic_mb' and 'internet_value' as the traffic column
    traffic_col = next(
        (c for c in ('traffic_mb', 'internet_value') if c in df.columns), None
    )

    base_date = None

    if 'timestamp' in df.columns and traffic_col is not None:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        # Aggregate to hourly if needed
        if len(df) > TOTAL_DAYS * HOURS_PER_DAY:
            df = df.set_index('timestamp')
            df = df[traffic_col].resample('1h').sum().reset_index()
        else:
            df = df.rename(columns={traffic_col: 'value'})
            df.columns = ['timestamp', 'value']
        traffic_col = df.columns[-1]
        values = df[traffic_col].values.astype(float)
        # Derive base_date from actual first timestamp (timezone-stripped)
        base_date = df['timestamp'].iloc[0].to_pydatetime().replace(tzinfo=None)
    else:
        # Single-column format: just the traffic values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError(
                f"Cannot find numeric traffic column in {path}. "
                f"Expected columns: timestamp, traffic_mb (or internet_value). "
                f"Got: {list(df.columns)}"
            )
        values = df[numeric_cols[0]].values.astype(float)

    expected = TOTAL_DAYS * HOURS_PER_DAY
    if len(values) < expected:
        raise ValueError(
            f"Dataset has only {len(values)} hourly values; "
            f"expected at least {expected} ({TOTAL_DAYS} days × {HOURS_PER_DAY}h)."
        )

    return values[:expected], base_date


def create_windows(values, base_date=None):
    """
    Create (x, y) sliding windows of length W and L respectively.
    Window t starts at hour t*HOURS_PER_DAY.
    Returns list of dicts: {timestep, x, y, date_start, date_end}
    """
    windows = []
    if base_date is None:
        base_date = datetime(2013, 11, 1)  # fallback

    for t in range(TOTAL_DAYS - 1):
        start = t * HOURS_PER_DAY
        if start + W + L > len(values):
            break
        x = values[start: start + W].tolist()
        y = values[start + W: start + W + L].tolist()
        windows.append({
            'timestep': t + 1,
            'x': x,
            'y': y,
            'date_start': (base_date + timedelta(days=t)).strftime('%Y-%m-%d'),
            'date_end':   (base_date + timedelta(days=t + 1)).strftime('%Y-%m-%d'),
        })

    return windows


def split_windows(windows):
    """Split into train (Days 1-21) and test (Days 22-30) windows."""
    train = [w for w in windows if w['timestep'] <= TRAIN_DAYS]
    test  = [w for w in windows if w['timestep'] >  TRAIN_DAYS]
    return train, test


def load_and_split(path=None):
    """Full pipeline: load → create windows → split."""
    values, base_date = load_data(path)
    windows = create_windows(values, base_date=base_date)
    train_windows, test_windows = split_windows(windows)
    return train_windows, test_windows, values


if __name__ == '__main__':
    train_windows, test_windows, values = load_and_split()

    print(f"Total hourly values : {len(values)}")
    print(f"Total windows       : {len(train_windows) + len(test_windows)}")
    print(f"Train windows       : {len(train_windows)}  (Days 1-{TRAIN_DAYS})")
    print(f"Test windows        : {len(test_windows)}   (Days {TRAIN_DAYS+1}-{TOTAL_DAYS-1})")
    print()
    w0 = train_windows[0]
    print(f"Sample window 1:")
    print(f"  date_start : {w0['date_start']}")
    print(f"  date_end   : {w0['date_end']}")
    print(f"  x (first 6): {w0['x'][:6]}")
    print(f"  y (first 6): {w0['y'][:6]}")
