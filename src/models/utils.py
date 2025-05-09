"""Useful functions for getting transformer language model surprisals."""

import functools
import math
import os
import random
import torch

from torch.nn.functional import softmax
from transformers import GPTNeoXForCausalLM, AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def organize_reading_materials(sentence_path, DATASETS): 

    ## goal is to end up concatenating all the sentences from each dataset
    all_sentences = []
    for dataset in DATASETS:
        file = dataset["name"]
        sheet_name = dataset.get("sheet_name", None)
        # print(f"File: {file}, Sheet Name: {sheet_name}")

        if "geco" in file:
            sentence_colname = "SENTENCE"
            sentenceid_colname = "SENTENCE_ID"
        elif "natstories" in file: 
            sentence_colname = "Sentence"
            sentenceid_colname = "SentNum"
        else:
            raise ValueError(f"Unknown dataset format in file: {file}")

        # Load the file
        file_path = os.path.join(sentence_path, file)
        if file.endswith(".xlsx"):
            sdf = pd.read_excel(file_path, sheet_name=sheet_name)
        elif file.endswith(".csv"):
            sdf = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file}")

        # Extract dataset name and build the DataFrame
        dataset_name = file.split("-")[0]
        temp_df = pd.DataFrame({
            "dataset_name": dataset_name,
            "sentence_number": sdf[sentenceid_colname],
            "sentences": sdf[sentence_colname]
        })
        all_sentences.append(temp_df)

    # Concatenate all datasets into a single DataFrame
    df = pd.concat(all_sentences, ignore_index=True)

    return df

def generate_revisions():
    ## TODO: Ensure this is correct
    # Fixed initial steps
    revisions = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    # Add every 1,000 steps afterward
    revisions.extend(range(2000, 144000, 1000))  # Adjust range as needed
    # Format each step as "stepX"
    return [f"step{step}" for step in revisions]

def find_sublist_index(mylist, sublist):
    """Find the first occurence of sublist in list.
    Return the start and end indices of sublist in list"""

    for i in range(len(mylist)):
        if mylist[i] == sublist[0] and mylist[i:i+len(sublist)] == sublist:
            return i, i+len(sublist)
    return None

@functools.lru_cache(maxsize=None)  # This will cache results, handy later...



def run_model(model, tokenizer, sentence, device):
    """Run model, return hidden states and attention"""
    # Tokenize sentence
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # Run model
    with torch.no_grad():
        output = model(**inputs, output_attentions=True)
        hidden_states = output.hidden_states
        attentions = output.attentions

    return {'hidden_states': hidden_states,
            'attentions': attentions,
            'tokens': inputs}

### ... grab the embeddings for your target tokens
def get_embedding(hidden_states, inputs, tokenizer, target, layer, device):
    """Extract embedding for TARGET from set of hidden states and token ids."""
    
    # Tokenize target
    target_enc = tokenizer.encode(target, return_tensors="pt",
                                  add_special_tokens=False).to(device)
    
    # Get indices of target in input tokens
    target_inds = find_sublist_index(
        inputs["input_ids"][0].tolist(),
        target_enc[0].tolist()
    )

    # Get layer
    selected_layer = hidden_states[layer][0]

    #grab just the embeddings for your target word's token(s)
    token_embeddings = selected_layer[target_inds[0]:target_inds[1]]

    #if a word is represented by >1 tokens, take mean
    #across the multiple tokens' embeddings
    embedding = torch.mean(token_embeddings, dim=0)
    
    return embedding

def count_parameters(model):
    """credit: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model"""
    
    total_params = 0
    for name, parameter in model.named_parameters():
        
        # if the param is not trainable, skip it
        if not parameter.requires_grad:
            continue
        
        # otherwise, count it towards your number of params
        params = parameter.numel()
        total_params += params
    # print(f"Total Trainable Params: {total_params}")
    
    return total_params

def compute_surprisal(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift logits and labels to align
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather log probs of the actual next tokens
    next_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Convert to surprisal: -log2(p)
    surprisals = -next_token_log_probs / math.log(2)

    tokens = tokenizer.convert_ids_to_tokens(shift_labels.squeeze().tolist())
    return list(zip(tokens, surprisals.squeeze().tolist()))
## example use: 
# text = "The quick brown fox jumps over the lazy dog"
# surprisal_output = compute_surprisal(text)

# for token, surprisal in surprisal_output:
#     print(f"Token: {token:<15} Surprisal: {surprisal:.4f}")

def compute_surprisal_batch(texts):
    # Tokenize with padding
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    next_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Convert to surprisal: -log2(p)
    surprisals = -next_token_log_probs / math.log(2)

    results = []
    for i, text in enumerate(texts):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i][1:shift_labels.shape[1] + 1])
        text_surprisals = surprisals[i][:len(tokens)]
        mask = shift_mask[i][:len(tokens)]
        # Filter out padding
        filtered = [(t, s.item()) for t, s, m in zip(tokens, text_surprisals, mask) if m.item() == 1]
        results.append(filtered)

    return results
## example use: 
# texts = ["The quick brown fox",
#          "GPT models can generate text.",
#          "Surprisal measures how unexpected a token is."]

# batched_output = compute_surprisal_batch(texts)

# for i, sentence in enumerate(texts):
#     print(f"\nInput: {sentence}")
#     for token, surprisal in batched_output[i]:
#         print(f"  Token: {token:<15} Surprisal: {surprisal:.4f}")
