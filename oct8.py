#import libraries 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict


#title of the project
st.title("""ClaimCatcher""")
st.subheader("Automated Medical Insurance Fraud Detection")
st.write("Welcome to ClaimCatcher. Please enter the details claim details.")

#fetch some data
#df = pd.read_csv("ClaimExport.csv")

@st.cache
def load_data(nrows):
    dfcopy = pd.read_csv("out.csv", nrows=nrows)
    lowercase = lambda x: str(x).lower()
    dfcopy.rename(lowercase, axis = "columns", inplace = True)
    return dfcopy


# Create a text element and let the reader know the data is loading.
#data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
dfcopy = load_data(10000)
# Notify the reader that the data was successfully loaded.
#data_load_state.text("Done! (using st.cache)")


#inspect the raw 
if st.checkbox('Click to see raw data?'):
    st. header('Raw data')
    st.write(dfcopy["total_amountusd"])

#draw a histogram
st.subheader('Amount claimed for each submitted claim')
st.bar_chart(dfcopy["total_amountusd"])

st.subheader('Amount approved for each submitted claim')
st.bar_chart(dfcopy["approved_amountusd"])
    
    
#input the date
date = st.sidebar.date_input(label='Input date of the claim')   

# User input on the enrollee id
enrollee_id1 = float(st.sidebar.text_input("Enter enrollee id", 0))

# Add a selectbox for HMO_ID
hmo_id1 = float(st.sidebar.selectbox(
    'What is the hmo id of the claim?',
    ('1', '2', '3'))
)

# User input on the enrollee id
provider_id1 = float(st.sidebar.text_input("Enter provider id", 0))


# User input on the care id
care_id1 = float(st.sidebar.text_input("Enter care id of the claim", 0))


#slider—quantity of the claim
claim_items_qty1 = st.sidebar.slider("Quantity of the claim", 0)
              
# slider—amount claimed per item
claim_items_amountusd1 = st.sidebar.slider('Amount claimed per item') 
st.sidebar.write("Total amount claimed is $", claim_items_amountusd1*claim_items_qty1)
total_amountusd1 = claim_items_amountusd1*claim_items_qty1

#claim_items_care_id
claim_items_care_id1 = float(st.sidebar.text_input('Claim items care id',0))

#claim_items_care_id
claim_items_id1 = float(st.sidebar.text_input('Claim items id',0))

#claim_items_care_id
claim_items_claim_id1 = float(st.sidebar.text_input('Claim items claim id',0))

##Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
rfc = RandomForestClassifier(n_estimators=100, random_state = 42)

X = dfcopy[["enrollee_id", 
            "hmo_id", 
            "provider_id",
            "care_id", 
             "claim_items_qty", 
            "claim_items_amountusd",
            "total_amountusd", 
            "claim_items_care_id", 
            "claim_items_id",
            "claim_items_claim_id"]]

y = dfcopy["hmo_status_-1.0"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rfc.fit(X_train,y_train)
rfc_predict = rfc.predict(X_test)
confusion_matrix(y_test, rfc_predict)

y_pre = rfc.predict(X_test)
cm = confusion_matrix(y_test, y_pre)
cm_display = ConfusionMatrixDisplay(cm).plot()

print(classification_report(y_test,y_pre))


####Take input from the user and plug that in the random forest model:
inputs = {
                'enrollee_id': enrollee_id1, 
                'hmo_id': hmo_id1, 
                'provider_id1': provider_id1, 
                'care_id': care_id1,
                "claim_items_qty": claim_items_qty1, 
                "claim_items_amountusd": claim_items_amountusd1,
                "total_amountusd": total_amountusd1,    
                "claim_items_care_id": claim_items_care_id1, 
                "claim_items_id": claim_items_id1,
                "claim_items_claim_id": claim_items_claim_id1
             }
#inputs

#Probability of this claim being fraudulent 
new_predictions = rfc.predict_proba([[enrollee_id1, 
            hmo_id1, 
            provider_id1,
            care_id1, 
            claim_items_qty1, 
            claim_items_amountusd1,
            total_amountusd1,    
            claim_items_care_id1, 
            claim_items_id1,
            claim_items_claim_id1]])
new_predictions

st.write ("Probability of this claim being fraudulent is:", new_predictions[:,1][0])

