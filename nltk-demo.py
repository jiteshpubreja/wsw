from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer

from nltk.stem import WordNetLemmatizer

from nltk.tag import pos_tag

from nltk.corpus import brown
from nltk.corpus import stopwords

text = "Do you know how NLTK works? It's actually quite interesting. Let's try something in this lecture. My name is Jitesh Pubreja and I'll be delivering this lecture."
words = ["writing","working","calves","wolves","branded","horse","randomize","possibly","provision","hospital","scratchy","kindness","Jitesh","Pubreja"]


def tag_list(tag):
    tlist = {"CC" : "conjunction, coordinating","CD" : "numeral, cardinal","DT" : "determiner","EX" : "existential there","IN" : "preposition or conjunction, subordinating","JJ" : "adjective or numeral, ordinal","JJR" : "adjective, comparative","JJS" : "adjective, superlative","LS" : "list item marker","MD" : "modal auxiliary","NN" : "noun, common, singular or mass","NNP" : "noun, proper, singular","NNS" : "noun, common, plural","PDT" : "pre-determiner","POS" : "genitive marker","PRP" : "pronoun, personal","PRP$" : "pronoun, possessive","RB" : "adverb","RBR" : "adverb, comparative","RBS" : "adverb, superlative","RP" : "particle","TO" : "'to' as preposition or infinitive marker","UH" : "interjection","VB" : "verb, base form","VBD" : "verb, past tense","VBG" : "verb, present participle or gerund","VBN" : "verb, past participle","VBP" : "verb, present tense, not 3rd person singular","VBZ" : "verb, present tense, 3rd person singular","WDT" : "WH-determiner","WP" : "WH-pronoun","WRB" : "Wh-adverb"}
    return tlist.get(tag,"Unknown Tag")

def chunker(input_words,n):
    words = input_words.split(" ")
    output = []

    current_chunk = []
    count = 0
    for w in words:
        current_chunk.append(w)
        count = count + 1
        if(count == n):
            output.append(current_chunk)
            current_chunk = []
            count = 0
    return output


stemmers = ['Porter','Lancaster','Snowball']
lemmatizers = ['WordNet Noun Lemmatizer','WordNet Verb Lemmatizer']

token_format = "{:>5} -> {:>1}"
stem_format = "{:>10} " + "{:>16} " * len(stemmers)
lemma_format = "{:>10} " + "{:>25} " * len(lemmatizers)
pos_format = "{:>20}{:>4}{:>10}\t{:>1}"

st = sent_tokenize(text)
wt = word_tokenize(text)
wpt = wordpunct_tokenize(text)

print("Sentence Tokenizer")
[print(token_format.format(x,y)) for x,y in enumerate(st)]
print()
print("Word Tokenizer")
print(wt)
print()
print("Word Punct Tokenizer")
print(wpt)
print()

print("Stemmers")
ps = PorterStemmer()
ls = LancasterStemmer()
ss = SnowballStemmer("english")

print("Languages Supported By Snowball Stemmer")
[print(x) for x in ss.languages]
print()

print(stem_format.format('Input',*stemmers))
print(stem_format.format('=' * 10,'=' * 16,'=' * 16,'=' * 16))
# [print(stem_format.format(x,ps.stem(x),ls.stem(x),ss.stem(x))) for x in wpt if len(x) > 1]
[print(stem_format.format(x,ps.stem(x),ls.stem(x),ss.stem(x))) for x in words]
print()

print("Stopwords Finder")
stop_words = set(stopwords.words('english'))
[print(x) for x in wpt if x in stop_words]
print()

print("Lemmatizers")
wnl = WordNetLemmatizer()

print(lemma_format.format('Input',*lemmatizers))
print(lemma_format.format('=' * 10,'=' * 25,'=' * 25,'=' * 25))
# [print(lemma_format.format(x,wnl.lemmatize(x),wnl.lemmatize(x,pos="v"))) for x in wpt if len(x) > 1]
[print(lemma_format.format(x,wnl.lemmatize(x),wnl.lemmatize(x,pos="v"))) for x in words]
print()



print("POS Tagging")
print(pos_format.format('Input',"",'POS Tag',"Meaning of POS Tag"))
print(pos_format.format('=' * 20,"",'=' * 10,'=' * 18))
[print(pos_format.format(x," -> ",y,tag_list(y))) for x,y in pos_tag(words)]
print()

print("Chunking")
[print(x) for x in chunker(text,3)]
print()

print(pos_format.format('Input',"",'POS Tag',"Meaning of POS Tag"))
print(pos_format.format('=' * 20,"",'=' * 10,'=' * 18))
[print(pos_format.format(x," -> ",y,tag_list(y))) for x,y in pos_tag(words)]
print()


print("POS Tagging Corpus")
print(pos_format.format('Input',"",'POS Tag',"Meaning of POS Tag"))
print(pos_format.format('=' * 20,"",'=' * 10,'=' * 18))
[print(pos_format.format(x," -> ",y,tag_list(y))) for x,y in pos_tag(brown.words()[:100]) if len(x) > 1]
print()