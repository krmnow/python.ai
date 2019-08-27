import os
import tarfile
from six.moves import urllib
import email
import email.policy
import numpy as np
from sklearn.model_selection import train_test_split
import re
from html import unescape

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("zestawy danych", "spam")


def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()


fetch_spam_data()

HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

print(len(ham_filenames))
print(len(spam_filenames))


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

print(ham_emails[4].get_content().strip())

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


from collections import Counter

def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures


print(structures_counter(ham_emails).most_common())


for header, value in spam_emails[0].items():
    print(header,":", value)

print(spam_emails[0]["Subject"])


X = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def html_to_text(html):
    text = re.sub('<head.*>.*?</head>', '', html, flags=re.M | re.S | re.I )
    text = re.sub('<a\s.*?>', ' HIPERŁĄCZE ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)

html_spam_emails = [email for email in X_train[y_train==1]:
                    if get_email_structure(email) == "tetx/html"]
sample_html_spam = html_spam_emails[3]
print(sample_html_spam.get_content().strip()[:1000], "...")
print(html_to_text(sample_html_spam.get_content())[:1000], "...")

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html")
            continue
        try:
            content = part.get_content()
        except: # w przypadku problemow z kodowaniem
            content = str(part.get_payload())
        if ctype == "tetx/plain":
            return content
        else:
            html = content

    if html:
        return html_to_text(html)

print(email_to_text(sample_html_spam)[:100], "...")


try:
    import nltk

    stemmer =nltk.PorterStemmer()
    for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
        print(word, "=>", stemmer.stem(word))
except ImportError:
    print("Blad: proces analizy slowotworczej wymaga modulu nltk")
    stemmer = None

try:
    import urlextract

    url_extractor = urlextract.URLExtract()
    print(url_extractor.find_urls("Czy wykryje tu lfc.pl i legia.net?"))
except ImportError:
    print("Blad: zastepowanie adresow url wymaga dostepnosci modulu urlextract")
    url_extractor = None



