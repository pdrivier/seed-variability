### Data wrangling to ensure that reading time data aligns with surprisals data


## Create a column in the surprisals dataframe for unique word identifiers
## The procedure will have to be unique for each corpus, since they are organized differently

import os
import utils

import numpy as np
import pandas as pd

reading_path = "../../data/raw/reading-times/"
materials_path = "../../data/raw/materials/"

reading_file = "geco-MonolingualReadingData.csv"
materials_file, sheet_name = ("geco-EnglishMaterials.xlsx","SENTENCE")

df_reading = pd.read_csv(os.path.join(reading_path,reading_file))
df_materials = pd.read_csv(os.path.join(materials_path,materials_file))

df_reading = utils.wrangle_metadata_geco(df_reading, df_materials)

# trim the reading dataframe to the metrics you'd like
# save to the organized reading materials