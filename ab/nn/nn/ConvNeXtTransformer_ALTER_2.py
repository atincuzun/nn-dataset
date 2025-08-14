self.decoder = TransformerDecoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_layers=num_layers,
    nhead=nhead,  # Change nhead here
    dropout=dropout,
    tie_weights=tie_weights
)