import csv
import os, sys, email
import numpy as np 
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf

#from keras.models import Sequential
#from keras.layers import Dense

#Reading the bad emails
file = open("PhishingEmails.csv",'r',encoding='utf8')
data = [row for row in csv.reader(file.read().splitlines())]
#data[1][0].encode('ascii','ignore').decode()
for i,liste in enumerate(data):
    for j,sublist in enumerate(liste):
        data[i][j] = data[i][j].encode("ascii","ignore").decode("ascii")
        
        
#reading the good emails
emails_df = pd.read_csv('GoodEmails.csv')
def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

def split_email_addresses(line):
    '''To separate multiple email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs


# Parse the good emails into a list email objects
messages = list(map(email.message_from_string, emails_df['message']))
emails_df.drop('message', axis=1, inplace=True)
keys = messages[0].keys()
for key in keys:
    emails_df[key] = [doc[key] for doc in messages]

# Parse content from good emails
emails_df['content'] = list(map(get_text_from_email, messages))

#Change columns names
df_goods = emails_df[['Subject','content']]
df_goods.columns = ['Subject','Body']


#Reorganizing the phishing emails data to match the good emails data
bad_emails = data
df_bad_emails = pd.DataFrame(bad_emails[1:],columns=bad_emails[0])

df_bads = df_bad_emails[['Subjecy','Body','Comments']]
df_bads['Body2'] = df_bads['Body'] + df_bads['Comments']
df_bads = df_bads[['Subjecy','Body2']]
df_bads.columns = ['Subject','Body']
#df_bads.head()


#features lists
features = ["SizeOfDocument","BodyNumChars","BodyNumWords","BodyNumUniqueWords","BodyRichness","BodyNumFunctionWords","BodyVerifyYourAccountPhrase" ,"SubjectNumChars","SubjectNumWords","SubjectRichness"]
words = ["able", "access", "account", "accounts", "action", "activate", "activated", "activation", "active", "activities", "admin", "administrator", "advise", "alert", "alerts", "allie", "allied", "allow", "alt", "alternative", "animal", "answered", "anti", "anything", "apologize", "asthma", "attach", "attached", "attachment", "attempt", "attention", "authorize", "aware", "away", "back", "balance", "bank", "banking", "banks", "banner", "believe", "best", "better", "beyond", "bill", "biz", "blackboard", "blog", "canada", "canopy", "captain", "card", "cards", "care", "cars", "carson", "cervices", "change", "changed", "check", "chief", "children", "choosing", "chord", "client", "click", "clicking", "college", "complete", "computer", "confirm", "confirmation", "continue", "convenient", "correct", "correctly", "could", "credit", "creek", "critical", "customer", "customers", "daly", "dangerous", "dari", "data", "days", "deactivate", "deactivated", "debit", "dear", "decline", "department", "depression", "description", "details", "different", "digital", "dir", "dire", "direct", "director", "disposition", "doctor", "document", "domain", "double", "due", "easy", "effective", "email", "emails", "ensure", "error", "even", "every", "everything", "executive", "expire", "failure", "fastest", "feature", "federally", "filelocker", "filename", "financing", "find", "finder", "first", "following", "food", "found", "frank", "franklin", "fraud", "freeze", "full", "function", "fwd", "gallery", "get", "good", "got", "gothic", "greg", "group", "growth", "hammad", "head", "head", "hello", "help", "helpful", "hen", "high", "hold", "honda", "however", "husky", "identify", "identity", "images", "immediately", "important", "inc", "includes", "inconvenience", "incorrect", "ind", "individuals", "information", "interruption", "invoicing", "issue", "itap", "kindly", "know", "latest", "lead", "learn", "legal", "let", "letting", "life", "like", "limit", "limited", "lin", "link", "locked", "lodge", "log", "logging", "logo", "logos", "long", "longer", "mai", "mail", "main", "maintenance", "make", "making", "mall", "man", "managing", "many", "marks", "may", "media", "medium", "member", "men", "message", "method", "minimum", "minutes", "miss", "mistake", "mobile", "mohammad", "monday", "monitored", "month", "monthly", "moodle", "much", "multiple", "need", "needed", "never", "new", "notice", "notification", "number", "officer", "often", "oil", "okay", "one", "operation", "pack", "page", "pain", "passion", "password", "paul", "pay", "paying", "payment", "payments", "people", "peoplesoft", "phone", "photo", "please", "point", "policy", "power", "present", "primary", "problem", "problems", "proceed", "process", "professional", "profile", "prompt", "protect", "protection", "proxy", "questions", "quick", "reach", "reader", "really", "reasons", "receive", "recently", "redirect", "reed", "register", "registered", "rel", "repayment", "reply", "request", "requested", "require", "reset", "resolve", "respond", "restore", "restrict", "reverse", "right", "riley", "risk", "roman", "rural", "safeguard", "safety", "said", "school", "scripts", "secure", "securely", "security", "see", "send", "sent", "serious", "server", "service", "services", "settings", "several", "sex", "shelter", "show", "signin", "sincerely", "site", "society", "social", "soft", "something", "standard", "statement", "still", "strong", "strongly", "subject", "super", "sure", "suspend", "suspended", "suspension", "tab", "take", "target", "team", "technical", "texts", "thank", "thanks", "think", "throat", "time", "times", "title", "today", "took", "toyota", "transfers", "treatment", "trust", "try", "two", "ufa", "ultimate", "unable", "unauthorized", "university", "unpaid", "unusual", "update", "updated", "upgraded", "uploads", "urgent", "use", "user", "username", "using", "validate", "value", "verification", "verify", "version", "via", "view", "voice", "want", "way", "website", "well", "wishes", "without", "work", "working", "world", "would"]
verbs = ["bring", "change", "check", "complete", "confirm", "create", "enter", "find", "give", "make", "open", "pay", "protect", "provide", "receive", "remove", "review", "sign", "update", "buy", "visit", "win", "delete", "approve", "set", "lose", "submit", "renew", "replace", "acquire", "obtain", "purchase", "click", "verify", "earn", "release", "share", "deposit", "activate", "reactivate", "reconfirm", "register", "download", "withdraw", "access", "assist", "fill", "secure", "validate", "deliver", "transfer", "discuss", "attach", "schedule", "raise", "build", "file", "consider", "reduce", "kill", "investigate"]

# Dataframes where to put the values of features in data
df_goods_features = pd.DataFrame([],columns=features)
df_goods_verbs = pd.DataFrame([],columns=verbs)
df_goods_words = pd.DataFrame([],columns=words)

df_bads_features = pd.DataFrame([],columns=features)
df_bads_verbs = pd.DataFrame([],columns=verbs)
df_bads_words = pd.DataFrame([],columns=words)


#Next 2 loops are for calculating the features values

# Note no need to execute this, it will take a lot of time; results stored in files, see below
#filling df_goods_features df_goods_verbs df_goods_words
for index,row in df_goods.iterrows():
    features_cnt = []
    doc = str(row['Subject'])+'\n'+str(row['Body'])
    
    verbs_cnt = [doc.count(verb) for verb in verbs]
    words_cnt = [doc.count(word) for word in words]
    
    #SizeOfDocument:Returns the email message size in bytes
    features_cnt.append(len(doc.encode('utf-8')))
    
    #BodyNumChars: Counts how many characters are in the email body
    body_char_cnt = sum(c.isalnum() for c in row['Body'])
    features_cnt.append(body_char_cnt)
    
    #BodyNumWords: Counts how many words are in the email body
    body_word_cnt = len(re.findall(r'\w+', row['Body']))
    features_cnt.append(body_word_cnt)
    
    #BodyNumUniqueWords:  Counts the number of unique words in the email body
    features_cnt.append(len(set(row['Body'])))
    
    #BodyRichness: Calculates the number of words divided by the number of characters found in the email body
    if sum(c.isalnum() for c in row['Body']) != 0:
        features_cnt.append( body_word_cnt / body_char_cnt )
    else:
        features_cnt.append(0)
    
    #BodyNumFunctionWords: Counts the number of function words discovered in the email body.
    features_cnt.append(sum(words_cnt))
    
    #BodyVerifyYourAccountPhrase: Binary feature that checks if the email body contains the words (verify your account)
    features_cnt.append(0 if 'verify your account' in doc.lower() else 1)
    
    #SubjectNumChars: Counts the number of characters the subject field contains and returns that number
    subject_char_cnt = sum(c.isalnum() for c in row['Subject'])
    features_cnt.append(subject_char_cnt)
    
    #SubjectNumWords: Counts the number of words the subject field contains and returns that number
    subject_word_cnt = len(re.findall(r'\w+', row['Subject']))
    features_cnt.append(subject_word_cnt)
    
    #SubjectRichness: Calculates the division between the number of words over the number of characters found in the subject field
    if sum(c.isalnum() for c in row['Subject']) != 0:
        features_cnt.append( subject_word_cnt / subject_char_cnt )
    else:
        features_cnt.append(0)
    df_goods_features.loc[len(df_goods_features)] = features_cnt
    df_goods_verbs.loc[len(df_goods_verbs)] = verbs_cnt
    df_goods_words.loc[len(df_goods_words)] = words_cnt
    print(index)
    


# Note: will take too long to execute the following; results stored in files, see below
#filling df_bads_features df_bads_verbs df_bads_words
for index,row in df_bads.iterrows():
    features_cnt = []
    doc = str(row['Subject'])+'\n'+str(row['Body'])
    
    verbs_cnt = [doc.count(verb) for verb in verbs]
    words_cnt = [doc.count(word) for word in words]
    
    #SizeOfDocument:Returns the email message size in bytes
    features_cnt.append(len(doc.encode('utf-8')))
    
    #BodyNumChars: Counts how many characters are in the email body 
    features_cnt.append(sum(c.isalnum() for c in str(row['Body'])))
    
    #BodyNumWords: Counts how many words are in the email body
    features_cnt.append(len(re.findall(r'\w+', str(row['Body']))))
    
    #BodyNumUniqueWords:  Counts the number of unique words in the email body
    features_cnt.append(len(set(str(row['Body']))))
    
    #BodyRichness: Calculates the number of words divided by the number of characters found in the email body
    if sum(c.isalnum() for c in str(row['Body'])) != 0:
        features_cnt.append( len(re.findall(r'\w+', str(row['Body']))) / sum(c.isalnum() for c in str(row['Body'])))
    else:
        features_cnt.append(0)
    
    #BodyNumFunctionWords: Counts the number of function words discovered in the email body.
    features_cnt.append(sum(words_cnt))
    
    #BodyVerifyYourAccountPhrase: Binary feature that checks if the email body contains the words (verify your account)
    features_cnt.append(0 if 'verify your account' in doc.lower() else 1)
    
    #SubjectNumChars: Counts the number of characters the subject field contains and returns that number
    features_cnt.append(sum(c.isalnum() for c in str(row['Subject'])))
    
    #SubjectNumWords: Counts the number of words the subject field contains and returns that number
    features_cnt.append(len(re.findall(r'\w+', str(row['Subject']))))
    
    #SubjectRichness: Calculates the division between the number of words over the number of characters found in the subject field
    if sum(c.isalnum() for c in str(row['Subject'])) != 0:
        features_cnt.append( len(re.findall(r'\w+', str(row['Subject']))) / sum(c.isalnum() for c in str(row['Subject'])))
    else:
        features_cnt.append(0)
    df_bads_features.loc[len(df_bads_features)] = features_cnt
    df_bads_verbs.loc[len(df_bads_verbs)] = verbs_cnt
    df_bads_words.loc[len(df_bads_words)] = words_cnt
    print(index)
    


#storing data into csv files
df_bads_verbs.to_csv(r'df_bads_verbs.csv',index = None, header=True)
df_bads_words.to_csv(r'df_bads_words.csv',index = None, header=True)
df_bads_features.to_csv(r'df_bads_features.csv',index = None, header=True)
df_bads.to_csv(r'df_bads.csv',index = None, header=True)

df_goods_verbs.to_csv(r'df_goods_verbs.csv',index = None, header=True)
df_goods_words.to_csv(r'df_goods_words.csv',index = None, header=True)
df_goods_features.to_csv(r'df_goods_features.csv',index = None, header=True)
df_goods.to_csv(r'df_goods.csv',index = None, header=True)


