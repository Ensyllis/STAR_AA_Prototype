import os
from typing import Dict, List
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import random

def load_and_preprocess_data(root_directory: str) -> Dict[str, List[str]]:
    texts_by_author = {
        'JK_Rowling': [],
        'JRR_Tolkien': []
    }
    
    print(f"Searching for files in: {root_directory}")
    for filename in os.listdir(root_directory):
        print(f"Found file: {filename}")
        if 'HarryPotter' in filename:
            with open(os.path.join(root_directory, filename), 'r', encoding='utf-8') as file:
                texts_by_author['JK_Rowling'].append(file.read())
        elif any(name in filename for name in ['LOTR', 'Hobbit', 'Silmarillion']):
            with open(os.path.join(root_directory, filename), 'r', encoding='utf-8') as file:
                texts_by_author['JRR_Tolkien'].append(file.read())
    
    print("Number of texts loaded:")
    for author, texts in texts_by_author.items():
        print(f"{author}: {len(texts)} texts")
    
    chunked_texts = {}
    for author, texts in texts_by_author.items():
        chunked_texts[author] = [text[i:i+512] for text in texts for i in range(0, len(text), 512)]
        print(f"Number of chunks for {author}: {len(chunked_texts[author])}")
    
    return chunked_texts

class AuthorDataset(Dataset):
    def __init__(self, chunked_texts: Dict[str, List[str]], tokenizer, max_len: int, views: int):
        self.texts = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.views = views
        
        for label, chunks in chunked_texts.items():
            self.texts.extend(chunks)
            self.labels.extend([label] * len(chunks))
        
        self.label_to_id = {label: i for i, label in enumerate(set(self.labels))}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.label_to_id[self.labels[idx]]
        
        # Create multiple views of the same text
        views = [text[i:i+self.max_len] for i in range(0, len(text), self.max_len)]
        views = views[:self.views]  # Limit to the specified number of views
        while len(views) < self.views:
            views.append(random.choice(views))  # Duplicate if not enough views
        
        encoded_views = [self.tokenizer.encode_plus(
            view,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        ) for view in views]
        
        input_ids = torch.cat([e['input_ids'] for e in encoded_views])
        attention_masks = torch.cat([e['attention_mask'] for e in encoded_views])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_dataloaders(chunked_texts: Dict[str, List[str]], tokenizer, batch_size: int, max_len: int, views: int):
    dataset = AuthorDataset(chunked_texts, tokenizer, max_len, views)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)