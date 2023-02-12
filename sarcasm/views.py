from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

data = pd.read_json("Sarcasm.json", lines=True)
data["is_sarcastic"] = data["is_sarcastic"].map({0: "Not Sarcasm", 1: "Sarcasm"})
data = data[["headline", "is_sarcastic"]]
x = np.array(data["headline"])
y = np.array(data["is_sarcastic"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = BernoulliNB()
model.fit(X_train, y_train)




def index(request):
   return render(request, 'index.html')

def about(request):
    return HttpResponse("<b>General Kenobi</b> <br> <a href = '/'> back </a>")

def analyze(request):
    djtext =request.POST.get('text', 'default')
    analyz = ""
    analyzed = ""
    for char in djtext:
            data = cv.transform([djtext]).toarray()
            analyz = model.predict(data)
    if(analyz == ['Sarcasm']):
                                analyzed = "Sarcasm"
    else:
        analyzed ="Not Sarcasm"                            
    params = {'purpose' : 'Detect Sarcasm', 'analyzed_text' : analyzed}
    return render(request, 'analyze.html', params)



