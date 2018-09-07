
# coding: utf-8

# # Naive Bayes Classifier

# In[43]:


import nltk 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
import string
import gensim


# In[62]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')


# In[63]:


print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))


# # Equation for Naive Bayes classifier

# ![aka.png](attachment:aka.png)

# In[44]:


with open('train.txt') as f:
    content = f.readlines()


# In[180]:


train_dict = {}
train_label_dict = {}
vocab = set()

for line in content:
    labels, sentence = line.split('\t',1)
    sentence = sentence.lower()
    # removing links
    sentence =  ' '.join([x for x in sentence.split() if ('<' not in x) and ('\\' not in x)])
    # removing numbers and punctuations
    sentence = ''.join([x for x in sentence if (x not in string.punctuation) and (not x.isdigit()) ])
    text = sentence.split()
    
    # remove the stop 
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords]
    for label in labels.split(','):
        label = label.strip()
        
        # for class probability
        if label not in train_label_dict:
            train_label_dict[label] = 1
        else:
            train_label_dict[label] += 1
            
        # for dictionary of each class
        if label not in train_dict:
            train_dict[label] = {}
        for w in text: # word count in each class
            if len(w)>3:
                if w not in train_dict[label]:
                    train_dict[label][w] = 1
                else:
                    train_dict[label][w] += 1 
            # adding to vocabulary 
            vocab.add(w)
            


# In[207]:


prior_prob= {}
total_class_label = sum(train_label_dict.values())
for key in train_label_dict.keys():
    prior_prob[key] = np.log(train_label_dict[key] / total_class_label)
prior_prob.values()


# In[204]:


vocab_size  = len(vocab)
vocab = list(vocab)
vocab_size


# In[205]:


# total words in class
for class_label in train_dict.keys():
    total_class_words = sum(train_dict[class_label].values())
    train_dict[class_label]['words_in_class'] = total_class_words


# In[206]:


train_dict['Main_Belt_asteroids']['words_in_class']


# In[208]:


i = 0 # correct predictions
m = 1
v = vocab_size
for line in content:
    labels, sentence = line.split('\t',1)
    sentence = sentence.lower()
    # removing links
    sentence =  ' '.join([x for x in sentence.split() if ('<' not in x) and ('\\' not in x)])
    # removing numbers and punctuations
    sentence = ''.join([x for x in sentence if (x not in string.punctuation) and (not x.isdigit()) ])
    
    text = sentence.split()
    # remove the stop 
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords]
    
    # removing blank space at the end of class label
    true_labels = [label.strip() for label in labels.split(',')]
    prob_den = {}
    for class_label in train_dict.keys():
        total_words_class =  train_dict[class_label]['words_in_class']
        for word in text:
            if (word in train_dict[class_label]) and len(word):
                count_fo_w_in_class = train_dict[class_label][word]
            else:
                count_fo_w_in_class = 0
            # probability distribution fo each word
            try:
                prob_den[class_label]=prob_den[class_label]+np.log((count_fo_w_in_class + m/v)/(total_words_class+1))
            except:
                prob_den[class_label]=np.log((count_fo_w_in_class + m/v)/(total_words_class+1))
        prob_den[class_label] = prob_den[class_label] + prior_prob[class_label]
    max_prob = -10000000
    for class_label in prob_den.keys():
        if(prob_den[class_label] >= max_prob):
            pred_class = class_label
            max_prob = prob_den[class_label]
    if(pred_class in true_labels):
        i= i+1
        print(true_labels," JGD ", pred_class)      

        
# In[229]:


train_accuracy = i / len(content) * 100
print("train accuracy of naive bayes classifier is = {}%".format(train_accuracy))


# In[212]:


i / len(content) * 100


# In[213]:


len(content)


# In[223]:


# reading the dev test dataset file 
with open('dev.txt') as f:
    content_dev = f.readlines()


# In[224]:


dev_corr = 0 # correct predictions
m = 1
v = vocab_size
for line in content_dev:
    labels, sentence = line.split('\t',1)
    sentence = sentence.lower()
    # removing links
    sentence =  ' '.join([x for x in sentence.split() if ('<' not in x) and ('\\' not in x)])
    # removing numbers and punctuations
    sentence = ''.join([x for x in sentence if (x not in string.punctuation) and (not x.isdigit()) ])
    
    text = sentence.split()
    # remove the stop 
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords]
    
    # removing blank space at the end of class label
    true_labels = [label.strip() for label in labels.split(',')]
    prob_den = {}
    for class_label in train_dict.keys():
        total_words_class =  train_dict[class_label]['words_in_class']
        for word in text:
            if (word in train_dict[class_label]) and len(word):
                count_fo_w_in_class = train_dict[class_label][word]
            else:
                count_fo_w_in_class = 0
            # probability distribution fo each word
            try:
                prob_den[class_label]=prob_den[class_label]+np.log((count_fo_w_in_class + m/v)/(total_words_class+1))
            except:
                prob_den[class_label]=np.log((count_fo_w_in_class + m/v)/(total_words_class+1))
        prob_den[class_label] = prob_den[class_label] + prior_prob[class_label]
    max_prob = -10000000
    for class_label in prob_den.keys():
        if(prob_den[class_label] >= max_prob):
            pred_class = class_label
            max_prob = prob_den[class_label]
    if(pred_class in true_labels):
        dev_corr = dev_corr +1
        print(true_labels," JGD ", pred_class)      


# In[228]:


dev_accuracy = dev_corr/len(content_dev)
print("Dev set accuracy of naive bayes classifier is= {}%".format(dev_accuracy*100))


# In[216]:


# reading the test dataset file file 
with open('test.txt') as f:
    content_test = f.readlines()


# In[217]:


test_corr = 0 # correct predictions
m = 1
v = vocab_size
for line in content_test:
    labels, sentence = line.split('\t',1)
    sentence = sentence.lower()
    # removing links
    sentence =  ' '.join([x for x in sentence.split() if ('<' not in x) and ('\\' not in x)])
    # removing numbers and punctuations
    sentence = ''.join([x for x in sentence if (x not in string.punctuation) and (not x.isdigit()) ])
    
    text = sentence.split()
    # remove the stop 
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords]
    
    # removing blank space at the end of class label
    true_labels = [label.strip() for label in labels.split(',')]
    prob_den = {}
    for class_label in train_dict.keys():
        total_words_class =  train_dict[class_label]['words_in_class']
        for word in text:
            if (word in train_dict[class_label]) and len(word):
                count_fo_w_in_class = train_dict[class_label][word]
            else:
                count_fo_w_in_class = 0
            # probability distribution fo each word
            try:
                prob_den[class_label]=prob_den[class_label]+np.log((count_fo_w_in_class + m/v)/(total_words_class+1))
            except:
                prob_den[class_label]=np.log((count_fo_w_in_class + m/v)/(total_words_class+1))
        prob_den[class_label] = prob_den[class_label] + prior_prob[class_label]
    max_prob = -10000000
    for class_label in prob_den.keys():
        if(prob_den[class_label] >= max_prob):
            pred_class = class_label
            max_prob = prob_den[class_label]
    if(pred_class in true_labels):
        test_corr = test_corr +1
        print(true_labels," JGD ", pred_class)      


# In[227]:


test_accuracy = test_corr/len(content_test)
print("test_accuracy of naive bayes classifieris= {}%".format(test_accuracy*100))

