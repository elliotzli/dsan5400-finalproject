import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import nltk
from nltk.corpus import stopwords


class GPT2Generator:
    def __init__(self):
        # Load the pre-trained GPT-2 model and tokenizer
        model_name = "gpt2-medium"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)


    def generate_reviews_forGPT(self,prompt, max_length=None, temperature=None):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

        # randomly generate hyperparameters
        max_length = np.random.randint(50, 200)
        temperature = np.random.uniform(0, 1.0)

        num_beams = int(np.random.randint(2, 10))
        no_repeat_ngram_size = int(np.random.randint(0, 5))

        # Set num_beam_groups to 1
        num_beam_groups = 1

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beam_groups=num_beam_groups,
        )

        generated_text = self.tokenizer.decode(output[0])
        return generated_text
    
    def clean_text(self,text):
        # Convert to lower cases and tokenize
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        text = ' '.join([word for word in text.split() if word not in (stopwords)])
        # Lemmatization
        lemmatizer = nltk.stem.WordNetLemmatizer()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text