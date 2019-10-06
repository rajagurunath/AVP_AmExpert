import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
st.title("AVP American Express Hackathon")
@st.cache
def read_data():
    train=pd.read_csv("../data/train.csv")
    compaign=pd.read_csv("../data/campaign_data.csv")
    coupon_item_mapping=pd.read_csv("../data/coupon_item_mapping.csv")
    customer_demographics=pd.read_csv("../data/customer_demographics.csv")
    customer_transaction=pd.read_csv("../data/customer_transaction_data.csv")
    item=pd.read_csv("../data/item_data.csv")
    test=pd.read_csv("../data/test_QyjYwdj.csv")
    return (train,compaign,coupon_item_mapping,
            customer_demographics,customer_transaction,
            item,test)

@st.cache
def merge_data(data_tuple):
    (train,compaign,coupon_item_mapping,
            customer_demographics,customer_transaction,
            item,test)=data_tuple
    def merge_helper(df):
        df=df.merge(compaign,how='left').\
                merge(customer_demographics,how='left')
        temp=coupon_item_mapping.merge(item, on='item_id',how='left')
        temp=temp.groupby(["coupon_id"])['item_id'].count()
        temp1=df.merge(temp,on="coupon_id",how='left')
        agg_cust_trans=customer_transaction.groupby("customer_id").sum()
        temp2=temp1.merge(agg_cust_trans,on='customer_id',how='left')
        return temp2
    train_feat=merge_helper(train)        
    assert train.shape[0]==train_feat.shape[0]
    test_feat=merge_helper(test)
    assert test.shape[0]==test_feat.shape[0]
    return train_feat,test_feat
@st.cache
def prepare_data():
    data_tuple =read_data()
    train,test=merge_data(data_tuple)
    return train,test

train,test=prepare_data()
st.sidebar.subheader("Analytics")
show_prepared_data=st.sidebar.checkbox("show merged data",False)
if show_prepared_data:
    st.title("Prepared data")
    st.subheader("Train")
    st.write(train.head(100))
    st.subheader("Test")
    st.write(test.head(100))

# st.line_chart(train['selling_price'])
# st.line_chart(train['other_discount'])
# st.line_chart(train['quantity'])
f=plt.figure()
plt.plot(train['selling_price'])
# st.plotly_chart(f)
train_1=train[train['redemption_status']==1]
# f=plt.figure()
plt.scatter(train_1.index,train_1['selling_price'],c='r')
plt.xlabel("data points")
plt.ylabel("selling price")
plt.title("selling price with redemption status")
st.plotly_chart(f)


f=plt.figure()
plt.scatter(train_1['coupon_discount'],train_1['selling_price'])
                        #c=train_1['marital_status'],
                        # s=train_1['family_size'])
plt.xlabel("coupon discount")
plt.ylabel("selling price")
plt.title("coupon discount vs selling price")

st.plotly_chart(f)




