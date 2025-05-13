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
    """Manually generate the list of checkpoints available for Pythia modeling suite"""
    
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
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens = True)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift logits and labels to align:
    # each row t of logits represents a prediction for label at position t+1
    # so: get rid of last row of logits (which has no corresponding actual label in labels sequence)
    # and: get ride of first item of labels, since there is no corresponding logit to predict it
    # see diagram below: x's are for eliminated elements in each variable
    # input labels: _x_ | __ | __ | __ 
    #                    /    /    /
    # outputlogits:  __ / __ / __ / _x_

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather log probs of the actual next tokens
    # (selects elements from log_probs along dim == 2, the vocabulary axis, using labels from
    # shift_labels -- unsqueeze(-1) gives shift_labels a third dimension)
    next_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Convert to surprisal: -log2(p) by invoking Change of Base Formula 
    # (the next_token_log_probs are in natural log, so we need to change to log_2 by dividing over log_e(2))
    surprisals = -next_token_log_probs / math.log(2)
    
    tokens = tokenizer.convert_ids_to_tokens(shift_labels.squeeze().tolist()) 
    token_surprisals = list(zip(tokens, surprisals.squeeze().tolist()))

    return token_surprisals
## example use: 
# text = "The quick brown fox jumps over the lazy dog"
# surprisal_output = compute_surprisal(text)

# for token, surprisal in surprisal_output:
#     print(f"Token: {token:<15} Surprisal: {surprisal:.4f}")

def clean_up_surprisals(token_surprisals): 

    words = []
    current_word = []
    current_surprisals = []

    # Combine surprisals (by summing) for words split into multiple subword tokens
    # Use the whitespace tokens to find word-initial segments
    for token, surprisal in token_surprisals:
        if " " in token and current_word:  # new word starts
            words.append(("".join(current_word), current_surprisals)) #save the last word
            current_word = [token] #reset for the new word
            current_surprisals = [surprisal]
        elif token.startswith("..."): #new word starts
            current_word.append(token)
            current_surprisals.append(surprisal)
        elif not token.startswith("...") and ("..." in current_word or "...." in current_word) and (not any(elem in token and current_word for elem in [".","!","?"])) : #add leading whitespace 
            if not " " in current_word:
                current_word.insert(0, " ") #prepend leading whitespace
                current_surprisals.insert(0,0) #prepend 0 correspondign to new whitespace
            current_word.append(token)
            current_surprisals.append(surprisal)
        elif any(elem in token and current_word for elem in [".","!","?"]):
            words.append(("".join(current_word), current_surprisals))
        else:
            current_word.append(token)
            current_surprisals.append(surprisal)

    # Remove the whitespace token surprisals
    # (this will also remove any words that don't have whitespaces in front of them, which 
    # takes care of anything that had been attached to the initial word in sentence
    # e.g. "I'm sure I will like it" --> "'m sure I will like it" --> gets rid of surprisal for " `m ")
    # e.g. "You, of course." --> ", of course" --> gets rid of surprisal for ","
    rmwh_surprisals = [(i.split()[0],j[1:]) for i,j in words if i.startswith(" ")]

    # Combine the surprisals for words with more than one subword token
    return [(i,np.sum(j)) for i,j in rmwh_surprisals]

