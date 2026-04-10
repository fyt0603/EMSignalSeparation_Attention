import h5py

with h5py.File("dataset/T0101_D00_S0111.mat", "r") as f:
    print(f"RF0_I shape: {f['RF0_I'].shape}")
    print(f"RF0_Q shape: {f['RF0_Q'].shape}")
    print(f"RF0_I shape: {f['RF0_I'].shape[1]}")
    print(f"RF0_Q shape: {f['RF0_Q'].shape[1]}")