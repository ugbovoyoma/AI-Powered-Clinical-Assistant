
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize


data= pd.read_csv(r'C:\Users\CB\Desktop\AI Powered App\data\raw\Symptom2Disease.csv')

data.head()

#Delete the column 'Unnamed: 0'
data= data.drop(columns=['Unnamed: 0'])

data


#Check for duplicates in the dataset

data.duplicated().sum()


#Drop duplicates
data= data.drop_duplicates()

data.duplicated().sum()

#Check the dataset for missing values

data.isnull().sum()

#Rename the columns

data= data.rename(columns={'label': 'diseases', 'text': 'symptoms'})

data['diseases']= data['diseases'].astype('str').str.strip()
data['symptoms']= data['symptoms'].astype('str').str.strip()

data.head()


#Check for rows with empty strings

(data['symptoms'] == '').sum(), (data['diseases'] == '').sum()


#Convert text to lowercase

data["symptoms"] = data["symptoms"].str.lower()

data.head()


# Create the folder if it doesn't exist
nltk.download('stopwords', download_dir=r'C:\Users\CB\nltk_data')
nltk.download('punkt',      download_dir=r'C:\Users\CB\nltk_data')
nltk.download('punkt_tab',  download_dir=r'C:\Users\CB\nltk_data')





nltk.data.path.append(r"C:\Users\CB\nltk_data")

# Try to load stopwords, fallback if missing
try:
    from nltk.corpus import stopwords
    STOP = set(stopwords.words("english"))
except Exception:
    STOP = {
        "the","and","is","in","to","of","a","it","that","for","on","with",
        "as","this","by","an","be","are","or","from","at","was","were","but"
    }

# Try to load the NLTK tokenizer, fallback if missing
try:
    from nltk.tokenize import word_tokenize
    _ = word_tokenize("test.") 
except Exception:
    def word_tokenize(text: str):
        return re.findall(r"\b\w+(?:'\w+)?\b", text)


# NLTK tokenization
data["tokens"] = data["symptoms"].apply(
    lambda s: [t for t in (tok.lower() for tok in word_tokenize(s))
               if any(c.isalpha() for c in t) and t not in STOP]
)

print("After clean:", data.shape)
data.head()

#Save cleaned data to a new CSV file

data.to_csv(r'C:\Users\CB\Desktop\AI Powered App\data\cleanedData\cleaned_symptom_disease.csv', index=False)

# Save as JSON
data.to_json(r'C:\Users\CB\Desktop\AI Powered App\data\cleanedData\cleaned_symptom_disease.json', orient='records', lines=True)


