# Oscillators

Small PyTorch script for forecasting on binary time-series data.

The model trains a GRU on the full observed history and predicts the next row by scoring every column and returning the top-k indices, where k is provided on the command line.

To avoid a built-in copy-the-last-row bias, the scorer uses a signed carryover term rather than forcing positive persistence from the previous row.
