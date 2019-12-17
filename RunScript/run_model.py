import re
import pandas as pd
import tensorflow
import numpy as np
from tensorflow.keras.models import model_from_json 
from sklearn import preprocessing
import sys
    
"""
usage:
    python run_model.py <input_csv>

see parameters below for configuration
"""


"""
PARAMETERS
"""

#fn = 'TestMLprediction.csv'
fn = sys.argv[1]
model_json = 'model.json'
model_weights = "model.h5"
output_name = 'testout.csv'
scaler_pkl_name = 'std_scale.pkl'
threshold = 0.7

"""
FUNCTIONS
"""
def load_model(model_json,model_weights):
    # load json and create model
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weights)
    print("Loaded model from disk")
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def make_data(filename):
    df_emails = pd.read_csv(filename)

    features = ["SizeOfDocument","BodyNumChars","BodyNumWords","BodyNumUniqueWords","BodyRichness","BodyNumFunctionWords","BodyVerifyYourAccountPhrase" ,"SubjectNumChars","SubjectNumWords","SubjectRichness"]
    words = ["able", "access", "account", "accounts", "action", "activate", "activated", "activation", "active", "activities", "admin", "administrator", "advise", "alert", "alerts", "allie", "allied", "allow", "alt", "alternative", "animal", "answered", "anti", "anything", "apologize", "asthma", "attach", "attached", "attachment", "attempt", "attention", "authorize", "aware", "away", "back", "balance", "bank", "banking", "banks", "banner", "believe", "best", "better", "beyond", "bill", "biz", "blackboard", "blog", "canada", "canopy", "captain", "card", "cards", "care", "cars", "carson", "cervices", "change", "changed", "check", "chief", "children", "choosing", "chord", "client", "click", "clicking", "college", "complete", "computer", "confirm", "confirmation", "continue", "convenient", "correct", "correctly", "could", "credit", "creek", "critical", "customer", "customers", "daly", "dangerous", "dari", "data", "days", "deactivate", "deactivated", "debit", "dear", "decline", "department", "depression", "description", "details", "different", "digital", "dir", "dire", "direct", "director", "disposition", "doctor", "document", "domain", "double", "due", "easy", "effective", "email", "emails", "ensure", "error", "even", "every", "everything", "executive", "expire", "failure", "fastest", "feature", "federally", "filelocker", "filename", "financing", "find", "finder", "first", "following", "food", "found", "frank", "franklin", "fraud", "freeze", "full", "function", "fwd", "gallery", "get", "good", "got", "gothic", "greg", "group", "growth", "hammad", "head", "head", "hello", "help", "helpful", "hen", "high", "hold", "honda", "however", "husky", "identify", "identity", "images", "immediately", "important", "inc", "includes", "inconvenience", "incorrect", "ind", "individuals", "information", "interruption", "invoicing", "issue", "itap", "kindly", "know", "latest", "lead", "learn", "legal", "let", "letting", "life", "like", "limit", "limited", "lin", "link", "locked", "lodge", "log", "logging", "logo", "logos", "long", "longer", "mai", "mail", "main", "maintenance", "make", "making", "mall", "man", "managing", "many", "marks", "may", "media", "medium", "member", "men", "message", "method", "minimum", "minutes", "miss", "mistake", "mobile", "mohammad", "monday", "monitored", "month", "monthly", "moodle", "much", "multiple", "need", "needed", "never", "new", "notice", "notification", "number", "officer", "often", "oil", "okay", "one", "operation", "pack", "page", "pain", "passion", "password", "paul", "pay", "paying", "payment", "payments", "people", "peoplesoft", "phone", "photo", "please", "point", "policy", "power", "present", "primary", "problem", "problems", "proceed", "process", "professional", "profile", "prompt", "protect", "protection", "proxy", "questions", "quick", "reach", "reader", "really", "reasons", "receive", "recently", "redirect", "reed", "register", "registered", "rel", "repayment", "reply", "request", "requested", "require", "reset", "resolve", "respond", "restore", "restrict", "reverse", "right", "riley", "risk", "roman", "rural", "safeguard", "safety", "said", "school", "scripts", "secure", "securely", "security", "see", "send", "sent", "serious", "server", "service", "services", "settings", "several", "sex", "shelter", "show", "signin", "sincerely", "site", "society", "social", "soft", "something", "standard", "statement", "still", "strong", "strongly", "subject", "super", "sure", "suspend", "suspended", "suspension", "tab", "take", "target", "team", "technical", "texts", "thank", "thanks", "think", "throat", "time", "times", "title", "today", "took", "toyota", "transfers", "treatment", "trust", "try", "two", "ufa", "ultimate", "unable", "unauthorized", "university", "unpaid", "unusual", "update", "updated", "upgraded", "uploads", "urgent", "use", "user", "username", "using", "validate", "value", "verification", "verify", "version", "via", "view", "voice", "want", "way", "website", "well", "wishes", "without", "work", "working", "world", "would"]
    verbs = ["bring", "change", "check", "complete", "confirm", "create", "enter", "find", "give", "make", "open", "pay", "protect", "provide", "receive", "remove", "review", "sign", "update", "buy", "visit", "win", "delete", "approve", "set", "lose", "submit", "renew", "replace", "acquire", "obtain", "purchase", "click", "verify", "earn", "release", "share", "deposit", "activate", "reactivate", "reconfirm", "register", "download", "withdraw", "access", "assist", "fill", "secure", "validate", "deliver", "transfer", "discuss", "attach", "schedule", "raise", "build", "file", "consider", "reduce", "kill", "investigate"]

    df_emails_features = pd.DataFrame([],columns=features)
    df_emails_verbs = pd.DataFrame([],columns=verbs)
    df_emails_words = pd.DataFrame([],columns=words)
    

    # Note no need to execute this, it will take a lot of time to execute, i stored the results in files as you will see further down
#filling df_emails_goods_features df_emails_goods_verbs df_emails_goods_words
    for index,row in df_emails.iterrows():
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
        df_emails_features.loc[len(df_emails_features)] = features_cnt
        df_emails_verbs.loc[len(df_emails_verbs)] = verbs_cnt
        df_emails_words.loc[len(df_emails_words)] = words_cnt
    #     print(index)

        # Adding 'v' to all verbs column names
    # Because they creat a conflict when joining verbs and words columns in same DataFrame
        df_emails_verbs.columns = ['v'+verb for verb in df_emails_verbs.columns]

        df_X = df_emails_features.join(df_emails_verbs).join(df_emails_words)
    return df_X
    
def run_model(dataframe_in,model):
    #normalizing the features values
    df = dataframe_in.copy()

    #Select numerical columns which needs to be normalized
    df_norm = df[df.columns[0:10]]

    # Normalize Training Data 
    # std_scale = preprocessing.StandardScaler().fit(data_norm)
    from sklearn.externals import joblib 
    #load standard-scaler from pickle file
    std_scale = joblib.load(scaler_pkl_name)
    df_transformed = std_scale.transform(df_norm)

    #Converting numpy array to dataframe
    norm_col = pd.DataFrame(df_transformed, index=df_norm.index, columns=df_norm.columns) 
    df.update(norm_col)

    model_prediction = []
    for i in range(len(df)):
        model_prediction.append(model.predict(df.iloc[[i]])) 

    model_prediction = np.array(model_prediction).reshape(len(model_prediction))
    
    binary_prediction = []
    for i in range(len(df)):
        if model_prediction[i]> threshold:
            binary_prediction.append(0)
        else: binary_prediction.append(1)
    
    df['binary_prediciton'] = binary_prediction
    df['model_prediciton'] = model_prediction
    return df

def write_out(df):
    df.to_csv(output_name,index = None, header=True)
    
"""
PROGRAM
"""

if __name__ == '__main__':
    m = load_model(model_json,model_weights)
    write_out(run_model(make_data(fn),m))
    