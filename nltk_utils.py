from nltk.stem.porter import PorterStemmer
import nltk
import numpy
nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = numpy.zeros(len(all_words), dtype=numpy.float32)

    for index, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[index] = 1.0

    return bag


a = "How long does shipping take?"
a = (tokenize(a))

print(a)

for i in a:
    print(stem(i))
