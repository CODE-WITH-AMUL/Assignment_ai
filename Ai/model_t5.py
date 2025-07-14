
import torch
import torch.nn as nn
import math
from transformers import T5Tokenizer
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

# Model Configuration Dictionary
model_config = {
    "model_type": "t5-like-encoder-decoder",
    "vocab_size": 32128,
    "d_model": 512,
    "d_ff": 2048,
    "num_layers": 6,
    "num_heads": 8,
    "dropout": 0.1,
    "max_position_embeddings": 512,
    "activation": "gelu",
    "relative_attention": True,
    "num_parameters_target": "7B"
}

# Relative Position Bias (T5-style)
class RelativePositionBias(nn.Module):
    def __init__(self, config, max_distance=128, bidirectional=True):
        super().__init__()
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.relative_attention_bias = nn.Embedding(32, config["num_heads"])
    
    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.relative_attention_bias.weight.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        if not self.bidirectional:
            relative_positions = relative_positions.clamp(min=0)
        relative_positions = relative_positions.clamp(-self.max_distance, self.max_distance)
        relative_positions += self.max_distance
        return self.relative_attention_bias(relative_positions)

# T5-like Model
class T5FromScratch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.positional_bias = RelativePositionBias(config) if config["relative_attention"] else None
        
        # Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["num_heads"],
            dim_feedforward=config["d_ff"],
            dropout=config["dropout"],
            activation=config["activation"]
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=config["num_layers"])
        
        # Decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=config["d_model"],
            nhead=config["num_heads"],
            dim_feedforward=config["d_ff"],
            dropout=config["dropout"],
            activation=config["activation"]
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=config["num_layers"])
        
        self.output_layer = nn.Linear(config["d_model"], config["vocab_size"])
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        # Embed inputs
        encoder_inputs = self.embedding(input_ids) * math.sqrt(self.config["d_model"])
        decoder_inputs = self.embedding(decoder_input_ids) * math.sqrt(self.config["d_model"])
        
        # Add relative position bias (simplified for demo)
        if self.config["relative_attention"]:
            seq_len = input_ids.size(1)
            rel_pos_bias = self.positional_bias(seq_len)
            # Note: In practice, add rel_pos_bias to attention scores (requires custom attention)
        
        # Encoder
        encoder_outputs = self.encoder(encoder_inputs, src_key_padding_mask=~attention_mask.bool())
        
        # Decoder
        decoder_outputs = self.decoder(
            decoder_inputs,
            encoder_outputs,
            tgt_key_padding_mask=~decoder_attention_mask.bool(),
            memory_key_padding_mask=~attention_mask.bool()
        )
        
        # Output logits
        logits = self.output_layer(decoder_outputs)
        return logits

# Training Loop
def train_model(model, tokenizer, dataset, epochs=3, batch_size=8, device="cuda"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataset:  # Assume dataset yields (input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
            input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            decoder_input_ids, decoder_attention_mask = decoder_input_ids.to(device), decoder_attention_mask.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
            loss = criterion(logits.view(-1, model.config["vocab_size"]), decoder_input_ids.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset)}")

# Example Usage
if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Use T5's tokenizer
    model = T5FromScratch(model_config)
    
    # Dummy dataset (replace with C4 or The Pile)
    sample_text = ["translate English to French: Hello world!", "summarize: This is a long text..."]
    inputs = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    decoder_inputs = tokenizer(["Bonjour le monde!", "Summary of the text."], padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    # Train (simplified)
    dataset = [(inputs["input_ids"], inputs["attention_mask"], decoder_inputs["input_ids"], decoder_inputs["attention_mask"])]
    train_model(model, tokenizer, dataset)
