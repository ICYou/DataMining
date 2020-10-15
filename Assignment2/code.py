import numpy as np
import pandas as pd
import os
import re
import string
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

def load_reviews(path, fold_nrs, label):
    columns = ['Raw', 'Processed', 'Label']
    df = pd.DataFrame()
    for i in fold_nrs:
        fold = "fold"+str(i)
        p = path + fold
        for file in os.listdir(p): # for each .txt file in this fold's folder
            if file.endswith(".txt"):
                f = open(os.path.join(p, file), "r")
                review = f.read()
                # remove whitespaces, numbers, punctuation, & make lowercase
                processed = process_string(review)
                new_row = pd.DataFrame([[review, processed, label]])
                df = df.append(new_row)
    df.columns = columns
    return df

def make_train_test_set():
    # Focusing only on the negative reviews
    path_dec = "./op_spam_v1.4/negative_polarity/deceptive_from_MTurk/"
    path_true = "./op_spam_v1.4/negative_polarity/truthful_from_Web/"

    # Label = 1 if it is a truthful (negative) review, =0 if it is a deceptive (negative) review

    #loading training set:
    train_dec = load_reviews(path_dec, np.arange(4)+1, 0) # folds 1-4 form the training set
    train_true = load_reviews(path_true, np.arange(4)+1, 1)
    train = pd.concat([train_dec, train_true])
    train = train.reset_index(drop=True)

    #loading the test set:
    test_dec = load_reviews(path_dec, [5], 0)  # test set for deceptive reviews
    test_true = load_reviews(path_true, [5], 1)
    test = pd.concat([test_dec, test_true])
    test = test.reset_index(drop=True)

    return [train,test]

def process_string(s):
    s = s.strip()    # remove whitespaces
    s = s.lower() # to lowercase
    s = re.sub(r'\d+', '', s) # remove numbers
    s = s.translate(str.maketrans("","", string.punctuation)) # remove punctuation
    return s




#t1="We stayed at the Schicago Hilton for 4 days and 3 nights for a conference. I have to say, normally I am very easy going about amenities, cleanliness, and the like...however our experience at the Hilton was so awful I am taking the time to actually write this review. Truly, DO NOT stay at this hotel. When we arrived in our room, it was clear that the carpet hadn't been vacuumed. I figuered, \"okay, it's just the carpet.\" Until I saw the bathroom! Although the bathroom had all the superficial indicators of housekeeping having recently cleaned (i.e., a paper band across the toilet, paper caps on the drinking glasses, etc., it was clear that no ACTUAL cleaning took place. There was a spot (probably urine!) on the toilet seat and, I kid you not, the remnants of a lip-smudge on the glass. I know people who have worked many years in the hotel industry and they always warned that lazy housekeeping will make things \"appear\" clean but in fact they make no effort to keep things sanitary. Well, the Hilton was proof. I called downstairs and complained, and they sent up a chambermaid hours later. Frankly, I found the room disgusting. The hotel itself, outside the rooms, was cavernous and unwelcoming, with an awful echo in the lobby area that created a migraine-inducing din. Rarely have I been so eager to leave a place as this. When I got home, I washed all my clothes whether I had worn them or not, such was the skeeviness of our accomodations. Please, do yourself a favor and stay at a CLEAN hotel."
#t2= "I stayed two nights at the Hilton Chicago. That was the last time I will be staying there. When I arrived, I could not believe that the hotel did not offer free parking. They wanted at least $10. What am I paying for when I stay there for the night? The website also touted the clean linens. The room was clean and I believe the linens were clean. The problem was with all of the down pillows etc. Don't they know that people have allergies? I also later found out that this hotel allows pets. I think that this was another part of my symptoms. If you like a clean hotel without having allergy attacks I suggest you opt for somewhere else to stay. I did not like how they nickel and dimed me in the end for parking. Beware hidden costs. I will try somewhere else in the future. Not worth the money or the sneezing all night."
#print(process_string(t2))


'''
train, test = make_train_test_set()
train.to_csv("./train.csv", header = ['Raw', 'Processed', 'Label'], index=False)
test.to_csv("./test.csv", header = ['Raw', 'Processed', 'Label'], index=False)
'''
train = pd.read_csv("./train.csv")
test = pd.read_csv("test.csv")

print(f"\n Shape of training set: {train.shape}")
print(train.head())
print(f"\n Shape of test set: {test.shape}")
print(test.head())
