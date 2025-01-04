import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


class Translator:
    def __init__(self, model_path, tokenizer_path, max_eng, max_fr):
        self.model = load_model(model_path)
        self.max_eng = max_eng
        self.max_fr = max_fr

        # Load tokenizers
        with open(tokenizer_path, "rb") as f:
            tokenizers = pickle.load(f)
            self.eng_tokenizer = tokenizers["eng_tokenizer"]
            self.fr_tokenizer = tokenizers["fr_tokenizer"]

        # Reverse mapping for French tokenizer
        self.idx_to_word = {idx: word for word, idx in self.fr_tokenizer.word_index.items()}
        self.idx_to_word[0] = "<PAD>"

    def translate(self, sentence):
        try:
            # Tokenize and pad the input sentence for the encoder
            encoder_input = self.eng_tokenizer.texts_to_sequences([sentence])
            encoder_input = pad_sequences(encoder_input, maxlen=self.max_eng, padding="post")

            # Initialize decoder input with the <start> token
            start_token_idx = self.fr_tokenizer.word_index.get("<start>", 1)
            decoder_input = np.zeros((1, self.max_fr - 1))  # Set shape to (1, max_fr - 1)
            decoder_input[0, 0] = start_token_idx

            # Translate word-by-word
            translated_sentence = []
            for i in range(1, self.max_fr - 1):  # Max length - 1 for decoder_input
                predictions = self.model.predict([encoder_input, decoder_input])
                predicted_word_idx = np.argmax(predictions[0, i - 1, :])
                predicted_word = self.idx_to_word.get(predicted_word_idx, "")

                if predicted_word == "<end>":
                    break

                if predicted_word != "<PAD>":  # Exclude <PAD> tokens
                    translated_sentence.append(predicted_word)

                decoder_input[0, i] = predicted_word_idx

            return " ".join(translated_sentence)
        except Exception as e:
            return f"Error during translation: {str(e)}"
