import nltk
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import cmudict
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv


# Step 0: Prepare the Data
nltk.download('cmudict')

merged_phonemes = [('AA', 'AO'), ('IH', 'EH'), ('IH', 'IY'), ('UH', 'UW'),
                   ('UH', 'OW'), ('UH', 'AH')]


# only the first rule wiht UH target will be applied below,
# as it will remove the target for the other goal phonemes:

context_sensitive_mergers = {
    ('V', '~', 'V', '/', '__', 'L:'): {'IH': 'IY', 'UH': 'UW', 'UH': 'OW', 'AH': 'AO', 'UH': 'AH'},
    ('V', '~', 'V', '/', '__', 'N'): {'IH': 'EH', "AA":"AO"}, #add dawn/don merger
    ('V', '~', 'V', '/', '__', 'R'): {'AO': 'AA'},
}


# Features
phoneme_features = {
    # Vowels
    "AA": ["vowel", "open", "back", "rounded"],
    "AE": ["vowel", "near-open", "front", "unrounded"],
    "AH": ["vowel", "open-mid", "back", "unrounded"],
    "AO": ["vowel", "open-mid", "back", "rounded"],
    "AW": ["vowel", "diphthong", "back", "rounded"],
    "AY": ["vowel", "diphthong", "front", "unrounded"],
    "EH": ["vowel", "open-mid", "front", "unrounded"],
    "ER": ["vowel", "rhotacized", "mid-central", "rounded"],
    "EY": ["vowel", "diphthong", "front", "unrounded"],
    "IH": ["vowel", "near-close", "front", "unrounded"],
    "IY": ["vowel", "close", "front", "unrounded"],
    "OW": ["vowel", "diphthong", "back", "rounded"],
    "OY": ["vowel", "diphthong", "front", "rounded"],
    "UH": ["vowel", "close", "back", "rounded"],
    "UW": ["vowel", "close", "back", "rounded"],

    # Consonants
    "B": ["consonant", "voiced", "bilabial", "plosive"],
    "CH": ["consonant", "voiceless", "post-alveolar", "affricate"],
    "D": ["consonant", "voiced", "alveolar", "plosive"],
    "DH": ["consonant", "voiced", "dental", "fricative"],
    "F": ["consonant", "voiceless", "labiodental", "fricative"],
    "G": ["consonant", "voiced", "velar", "plosive"],
    "HH": ["consonant", "voiceless", "glottal", "fricative"],
    "JH": ["consonant", "voiced", "post-alveolar", "affricate"],
    "K": ["consonant", "voiceless", "velar", "plosive"],
    "L": ["consonant", "voiced", "alveolar", "approximant"],
    "M": ["consonant", "voiced", "bilabial", "nasal"],
    "N": ["consonant", "voiced", "alveolar", "nasal"],
    "NG": ["consonant", "voiced", "velar", "nasal"],
    "P": ["consonant", "voiceless", "bilabial", "plosive"],
    "R": ["consonant", "voiced", "alveolar", "approximant"],
    "S": ["consonant", "voiceless", "alveolar", "fricative"],
    "SH": ["consonant", "voiceless", "post-alveolar", "fricative"],
    "T": ["consonant", "voiceless", "alveolar", "plosive"],
    "TH": ["consonant", "voiceless", "dental", "fricative"],
    "V": ["consonant", "voiced", "labiodental", "fricative"],
    "W": ["consonant", "voiced", "bilabial", "approximant"],
    "WH": ["consonant", "voiceless", "bilabial", "approximant"],
    "Y": ["consonant", "voiced", "palatal", "approximant"],
    "Z": ["consonant", "voiced", "alveolar", "fricative"],
    "ZH": ["consonant", "voiced", "post-alveolar", "fricative"]
}




def extract_features(phoneme):
    features = []
    #for p in phoneme:
    #print("Original Phoneme with Stress:", phoneme)
    if re.match(r"[A-Z]+\d*", phoneme):
        p = re.sub(r"\d+", "", phoneme)  # Remove stress indicator
    #print("Phoneme without Stress:", p)

    if p in phoneme_features:
        features += phoneme_features[p]

    return features



def context_sensitive_merge_condition(phoneme, i):
  #  print("csmc: ", phoneme, " i: ", i)
    # Check if the current phoneme is the goal of a context-sensitive merger
    for context, replacements in context_sensitive_mergers.items():
   #     print("context: ", context, " replacements: ", replacements)
        # Check if the current phoneme is a goal
    #    print("goals: ", replacements.keys())
        if phoneme[i] in replacements.keys():

            # Check if the previous and next phonemes satisfy the context
            prev_phoneme = phoneme[i - 1] if i > 0 else None
            next_phoneme = phoneme[i + 1] if i < len(phoneme) - 1 else None
            if (
                prev_phoneme == context[3]
                and next_phoneme == context[5]
            ):
                return True
    return False



# Step 1 - create base CMU dataframe
base_data = []
for word, phoneme_variants in cmudict.dict().items():
    # Take the first pronunciation variant
    phoneme = phoneme_variants[0]

    # Remove stress indicators from phonemes
    phoneme = [re.sub(r"\d+", "", p) for p in phoneme]

    base_data.append({'word': word, 'phoneme': phoneme, 'merge_status': 0, 'original_phoneme': phoneme.copy(), "merger_phones": 'None'})

base_df = pd.DataFrame(base_data)
#print("Base dataframe:")
#print(base_df.head())
base_df.to_csv('baseDF1.csv', index=False)



# Step 2: Make a context-free merged DataFrame
# Make a copy of base_df before iterating over phoneme pairs
df_combined = base_df.copy()

for goal_phoneme, target_phoneme in merged_phonemes:
    for i, row in df_combined.iterrows():
        phoneme = row['phoneme'].copy()  # Make a copy to prevent modifying the original data while iterating

        if goal_phoneme in phoneme and target_phoneme not in phoneme:
            df_combined.at[i, 'merge_status'] = 1
            df_combined.at[i, 'merger_phones'] = f'CF: {goal_phoneme} -> {target_phoneme}'


        # If the phoneme list contains the target phoneme, change it to the goal phoneme
        for j, p in enumerate(phoneme):
            if p == target_phoneme:
                phoneme[j] = goal_phoneme
                df_combined.at[i, 'merge_status'] = 2
                df_combined.at[i, 'merger_phones'] = f'CF: {goal_phoneme} -> {target_phoneme}'

        df_combined.at[i, 'phoneme'] = phoneme  # Assign the modified phoneme list back to the DataFrame

df_combined.to_csv('combinedDF1.csv', index=False)


# Step 3: Apply context-sensitive merges
context_sensitive_df = base_df.copy()  # Create a copy of base_df

for context, replacements in context_sensitive_mergers.items():
    for goal_phoneme, target_phoneme in replacements.items():
        for i, row in context_sensitive_df.iterrows():
            phoneme = row['phoneme'].copy()  # Make a copy to prevent modifying the original data while iterating

            original_phoneme = row['original_phoneme'].copy()  # Make a copy to store the original phoneme list

            # If the phoneme list contains the goal phoneme and not the target, set merge_status to 1
            if goal_phoneme in phoneme and target_phoneme not in phoneme:
                context_sensitive_df.at[i, 'merge_status'] = 1
                context_sensitive_df.at[i, 'merger_phones'] = f'CS: {goal_phoneme} -> {target_phoneme}'

            # If the phoneme list contains the target phoneme and the context is right, change it to the goal phoneme
            for j, p in enumerate(phoneme):
                if p == target_phoneme and context_sensitive_merge_condition(phoneme, j):
                    phoneme[j] = goal_phoneme
                    context_sensitive_df.at[i, 'merge_status'] = 2
                    context_sensitive_df.at[i, 'merger_phones'] = f'CS: {goal_phoneme} -> {target_phoneme}'

            context_sensitive_df.at[i, 'phoneme'] = phoneme  # Assign the modified phoneme list back to the dataframe

context_sensitive_df.to_csv('contextualDF1.csv', index=False)

# Step 4: Combine the two dataframes, overwriting the merge status and phoneme values from context-sensitive dataframe
merged_df = df_combined.copy()
for i, row in context_sensitive_df.iterrows():
    if row['merge_status'] == 2:
        merged_df.at[i, 'merge_status'] = row['merge_status']
        merged_df.at[i, 'phoneme'] = row['phoneme']
        merged_df.at[i,'merger_phones'] = row['merger_phones']

merged_df.to_csv('mergedDF1.csv', index=False)





def unmerged_target(phoneme_list):
    def get_stress(phoneme):
        return int(phoneme[-1]) if phoneme[-1].isdigit() else -1

    phoneme_with_max_stress = max(phoneme_list, key=get_stress, default=None)

    if phoneme_with_max_stress is None:
        return None, None

    return phoneme_with_max_stress[:-1], phoneme_list.index(phoneme_with_max_stress)



#Step 5: Context!
columns = ["target_feature1", "target_feature2", "target_feature3", "target_feature4",
           "goal_feature1", "goal_feature2", "goal_feature3", "goal_feature4",
           "preceding_feature1", "preceding_feature2", "preceding_feature3", "preceding_feature4",
           "following_feature1", "following_feature2", "following_feature3", "following_feature4"]

context_count = 0
def get_context(row):
    global context_count
    context_count += 1


    phonemes = row['phoneme']
    original_phonemes = row['original_phoneme']
    target_index = None
    target_features, goal_features, preceding_features, following_features = ['#']*4, ['#']*4, ['#']*4, ['#']*4

    if row['merge_status'] == 0:
        target_phoneme, target_index = unmerged_target(cmudict.dict()[row["word"]][0])
        goal_phoneme = target_phoneme
    else: # row['merge_status'] == 2:
        for goal_phoneme, target_phoneme in merged_phonemes:
            if goal_phoneme in phonemes:
                target_index = phonemes.index(goal_phoneme)
                target_phoneme = original_phonemes[target_index]
                break
##    else:
##        print(f"Unexpected merge_status: {row['merge_status']}")
##        return target_features + goal_features + preceding_features + following_features

    preceding_phoneme = phonemes[target_index-1] if target_index > 0 else None
    following_phoneme = phonemes[target_index+1] if target_index < len(phonemes) - 1 else None


    missing_value = ['#','#','#','#']
    # Handle missing keys in the phoneme_features dictionary
    target_features[:len(phoneme_features.get(target_phoneme, []))] = phoneme_features.get(target_phoneme, [])[:4]
    goal_features[:len(phoneme_features.get(goal_phoneme, []))] = phoneme_features.get(goal_phoneme, [])[:4]
    if preceding_phoneme:
        preceding_features[:len(phoneme_features.get(preceding_phoneme, []))] = phoneme_features.get(preceding_phoneme, [])[:4]
    else:
        preceding_features = missing_value

    if following_phoneme:
        following_features[:len(phoneme_features.get(following_phoneme, []))] = phoneme_features.get(following_phoneme, [])[:4]
    else:
        following_features = missing_value
    output = target_features + goal_features + preceding_features + following_features
    if context_count % 5000 == 0:
        print(f"Analyzed: {context_count} words!")
        print("word: ", row['word'])
        print("output: ", output)
        print("output length: ", len(output))
    if len(output) != 16:
        print(row['word'], " is raising an Error!!!!")
        print("output: ", output)
        print("output length: ", len(output))
        return missing_value + missing_value + missing_value + missing_value
        raise ValueError("Output list is not of length 16")

    else:
        return output

# Use the apply function to apply this function across the DataFrame
context_df = merged_df.apply(lambda row: get_context(row), axis=1, result_type='expand')
context_df.columns = columns

# Join the context_df with the original DataFrame (merged_df)
merged_df = pd.merge(merged_df, context_df, left_index=True, right_index=True)

merged_df.to_csv("finalDF3.csv", index = False)





