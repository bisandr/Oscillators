# Binary Oscillator Forecaster

Small PyTorch script for one-step-ahead forecasting on binary time-series data with a fixed number of active entries per row.

The model trains a GRU on the full observed history and predicts the next row by scoring every column and returning the top-k indices, where k is provided on the command line.

To avoid a built-in copy-the-last-row bias, the scorer uses a signed carryover term rather than forcing positive persistence from the previous row.

## What It Supports

- CSV input with no header.
- Variable sequence length `T`.
- Column count read directly from the CSV file.
- Configurable number of picks from the command line.

## Typical Use Cases

- Forecast the next active components in the original `T x 50` dataset with `--picks 5`.
- Test the pipeline quickly on the included `T x 12` sample file with `--picks 2`.
- Validate that a new CSV matches a required active-count rule before using it for forecasting.
- Reuse the same script on files with different widths without encoding the width in the CLI.

## Common Commands

Run the original 50-column case:

```bash
python binary_oscillator.py data.csv --picks 5
```

Run the included 12-column sample:

```bash
python binary_oscillator.py sample_data_12.csv --picks 2
```

Use a custom pick count on another file width:

```bash
python binary_oscillator.py my_data.csv --picks 4
```

## Output

The script prints:

- the loaded shape `T, N`
- training progress during optimization, including the learned carryover value
- the predicted top-k indices for the next row in both 0-based and 1-based form

## Files

- `binary_oscillator.py`: training and forecasting script
- `sample_data_12.csv`: small sample dataset for quick verification
- `summary.md`: background notes on the modeling approach