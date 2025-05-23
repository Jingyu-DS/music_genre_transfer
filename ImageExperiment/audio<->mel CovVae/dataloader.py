def get_data_loaders(X, y, batch_size, test_size=0.1, seed=42):
    idx_tr, idx_va = train_test_split(
        np.arange(len(X)), test_size=test_size,
        stratify=y, random_state=seed
    )
    train_ds = TensorDataset(X[idx_tr], y[idx_tr])
    val_ds   = TensorDataset(X[idx_va], y[idx_va])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
