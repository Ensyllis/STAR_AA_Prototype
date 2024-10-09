import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from transformers import AutoTokenizer

from data_processing import load_and_preprocess_data, create_dataloaders
from model_definition import AuthorshipModel

# Hyperparameters
BATCH_SIZE = 32
MAX_LEN = 512
VIEWS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
MODEL_NAME = 'roberta-base'

# Load and preprocess data
root_directory = './data'
chunked_texts = load_and_preprocess_data(root_directory)

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create dataloaders
train_dataloader = create_dataloaders(chunked_texts, tokenizer, BATCH_SIZE, MAX_LEN, VIEWS)

# Initialize model
model = AuthorshipModel(MODEL_NAME, num_classes=len(chunked_texts), learning_rate=LEARNING_RATE)

# Setup training
checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', filename='author-model-{epoch:02d}-{val_loss:.2f}', save_top_k=3, monitor='val_loss')
logger = TensorBoardLogger("tb_logs", name="author_model")

trainer = Trainer(
    max_epochs=NUM_EPOCHS,
    callbacks=[checkpoint_callback],
    logger=logger,
    gpus=1 if torch.cuda.is_available() else 0
)

# Train the model
trainer.fit(model, train_dataloader)

print("Training completed!")

# Save the final model
trainer.save_checkpoint("author_model_final.ckpt")
print("Model saved as author_model_final.ckpt")