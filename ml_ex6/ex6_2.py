import re
from nltk.stem import PorterStemmer


def vocab_dict(list):
    dict = {}
    for i in list:
        value, key = i.split("\t")[:]
        dict[key] = value
    return dict


def process_email(email, vocab):
    email = email.lower()     # lower case
    email = re.sub('http://[^\s]*' or 'https://[^\s]*', 'httpaddr',  email)     # normalizing URLs
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)     # normalizing email addresses
    email = re.sub('[0-9]+', 'number', email)     # normalizing numbers
    email = re.sub('[$]+', 'dollar', email)     # normalizing dollars

    # strip all special characters
    special_char = ["<", "[", "^", ">", "+", "?", "!", "'", ".", ",", ":"]
    for char in special_char:
        email = email.replace(str(char), "")
        email = email.replace("\n", " ")

    # stem the word
    ps = PorterStemmer()
    email = [ps.stem(token) for token in email.split(" ")]
    email = " ".join(email)

    # process the email and return word_indices
    word_indices = []
    for char in email.split():
        if len(char) > 1 and char in vocab:
            word_indices.append(int(vocab[char]))
    return word_indices


def main():
    file_contents = open('emailSample1.txt', 'r').read()
    vocab_list = open('vocab.txt', 'r').read()

    vocab_list = vocab_list.split('\n')[:-1]
    vocab = vocab_dict(vocab_list)
    word_indices = process_email(file_contents, vocab)


main()
