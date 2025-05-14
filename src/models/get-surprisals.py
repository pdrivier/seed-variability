## Grab Pythia surprisals for words in passages from eyetracking reading experiment


import os
import torch
import transformers
import utils

import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer

## List all models
MODELS = [
         'EleutherAI/pythia-14m',
         # 'EleutherAI/pythia-70m',
         # 'EleutherAI/pythia-160m',
         #  'EleutherAI/pythia-410m',
          # 'EleutherAI/pythia-1b',
          # 'EleutherAI/pythia-1.4b',
          # 'EleutherAI/pythia-2.8b',
          # 'EleutherAI/pythia-6.9b',
          # 'EleutherAI/pythia-12b',
          ]

## Define path to sentences
sentence_path = "../../data/raw/"
DATASETS = [
    {"name": "geco-EnglishMaterials.xlsx", "sheet_name": "SENTENCE", "sentence_colname": "SENTENCE"},
    {"name": "natstories-parsed-natural-stories.xlsx", "sheet_name": "Boar", "sentence_colname": "SENTENCE"}
]


### Handle logic for a dataset/model
def main(df, mpath, revisions, cachepath):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))

    for checkpoint in tqdm(revisions):

        print(checkpoint)
        for seed in range(1, 10):

            seed_name = "seed" + str(seed)

            model_name = mpath + "-" + seed_name
            print(model_name)

            ### Set up save path, filename, etc.
            savepath = "../../data/processed/"
            if not os.path.exists(savepath): 
                os.mkdir(savepath)
            if "/" in mpath:
                filename = "surprisals-model-" + mpath.split("/")[1] + "-" + seed_name + "-" + checkpoint + ".csv"
            else:
                filename = "surprisals-model-" + mpath + "-" + seed_name + "-" + checkpoint + ".csv"

            print("Checking if we've already run this analysis...")
            if os.path.exists(os.path.join(savepath,filename)):
               print("Already run this model for this checkpoint.")
               continue

            print("Loading model")
            ### if it doesn't exist, run it.
            print(model_name)
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=checkpoint,
                output_hidden_states = True
            )
            model.to(device) # allocate model to desired device

            tokenizer = AutoTokenizer.from_pretrained(mpath, revision=checkpoint)

            n_layers = model.config.num_hidden_layers
            print("number of layers:", n_layers)
            n_heads = model.config.num_attention_heads
            print("number of heads:", n_heads)
        
            n_params = utils.count_parameters(model)
        
            results = []

            ## Set up code to grab surprisals below
            # would be more efficient if set up in batches but alas
            for ix, row in tqdm(df.iterrows(),total=len(df)):
                
                # Load the current sentence
                sentence = row["sentences"]
                dataset_name = row["dataset_name"]

                # Compute and clean up surprisals
                token_surprisals = utils.compute_surprisal(sentence,tokenizer,model,device)

                if len(token_surprisals) > 1:
                    
                    word_surprisals = utils.clean_up_surprisals(token_surprisals, dataset_name)

                    
                    for word, surprisal in word_surprisals:
                        ### Add to results dictionary
                        results.append({
                            "dataset_name": dataset_name,
                            'sentence': sentence,
                            'word': word,
                            'surprisal': surprisal
                        })

                #TODO: see if you can polish the below to get surprisals for batched sentences
                # BATCH_SIZE = 16
                # batched_sentences = []
                # meta = []

                # for _, row in df.iterrows():
                #     sentence = row["sentences"]
                #     dataset_name = row["dataset_name"]
                    
                #     batched_sentences.append(sentence)
                #     meta.append((sentence, dataset_name))
                    
                #     if len(batched_sentences) >= BATCH_SIZE:
                #         results_batch = batch_compute_surprisal(batched_sentences, model, tokenizer, device)

                #         for (sentence, dataset_name), word_surprisals in zip(meta, results_batch):
                #             for word, surprisal in word_surprisals:
                #                 results.append({
                #                     "dataset_name": dataset_name,
                #                     "sentence": sentence,
                #                     "word": word,
                #                     "surprisal": surprisal
                #                 })

                #         batched_sentences = []
                #         meta = []



        
            df_results = pd.DataFrame(results)
            df_results['model'] = mpath.split("/")[1]
            df_results['revision'] = checkpoint
            df_results['seed_name'] = seed_name
            df_results['seed'] = seed
            df_results['step'] = int(checkpoint.replace("step", ""))
            df_results["n_params"] = n_params

            if cachepath: 
                utils.clear_model_from_cache(cachepath)




if __name__ == "__main__":

    ## Specify the model cache, so you can delete models after each run
    cachepath = "../../../../../../.cache/huggingface/hub/"

    ## Read stimuli
    df = utils.organize_reading_materials(sentence_path,DATASETS)
    filename = "all-sentences.csv"
    savedfpath = sentence_path + "organized_reading_materials/"
    if not os.path.exists(savedfpath):
        os.mkdir(savedfpath)
    df.to_csv(os.path.join(savedfpath,filename))

    ### Get model checkpoints/revisions
    revisions = utils.generate_revisions()

    ## Run main
    main(df, MODELS[0], revisions, cachepath)



