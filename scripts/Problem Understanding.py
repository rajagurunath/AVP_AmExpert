#!/usr/bin/env python
# coding: utf-8

# ## Predicting Coupon Redemption
# XYZ Credit Card company regularly helps it’s merchants understand their data better and take key business decisions accurately by providing machine learning and analytics consulting. ABC is an established Brick & Mortar retailer that frequently conducts marketing campaigns for its diverse product range. As a merchant of XYZ, they have sought XYZ to assist them in their discount marketing process using the power of machine learning. Can you wear the AmExpert hat and help out ABC?
# 
#  
# Discount marketing and coupon usage are very widely used promotional techniques to attract new customers and to retain & reinforce loyalty of existing customers. The measurement of a consumer’s propensity towards coupon usage and the prediction of the redemption behaviour are crucial parameters in assessing the effectiveness of a marketing campaign.
# 
#  
# ABC’s promotions are shared across various channels including email, notifications, etc. A number of these campaigns include coupon discounts that are offered for a specific product/range of products. The retailer would like the ability to predict whether customers redeem the coupons received across channels, which will enable the retailer’s marketing team to accurately design coupon construct, and develop more precise and targeted marketing strategies.
# 
#  
# The data available in this problem contains the following information, including the details of a sample of campaigns and coupons used in previous campaigns -
# 
# User Demographic Details
# Campaign and coupon Details
# Product details
# Previous transactions
# Based on previous transaction & performance data from the last 18 campaigns, predict the probability for the next 10 campaigns in the test set for each coupon and customer combination, whether the customer will redeem the coupon or not?
# 
#  
# 
# 

# ## Dataset Description
# Here is the schema for the different data tables available. The detailed data dictionary is provided next.
# 

# ![](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2019/09/Screenshot-2019-09-28-at-8.58.32-PM.png)

# ## To summarise the entire process:
# 
# Customers receive coupons under various campaigns and may choose to redeem it.
# They can redeem the given coupon for any valid product for that coupon as per coupon item mapping within the duration between campaign start date and end date
# Next, the customer will redeem the coupon for an item at the retailer store and that will reflect in the transaction table in the column coupon_discount.
#  

# In[15]:



import featuretools as ft
import pandas as pd
import numpy as np
from featuretools_tsfresh_primitives import AbsEnergy
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


train=pd.read_csv("../data/train.csv")
compaign=pd.read_csv("../data/campaign_data.csv")
coupon_item_mapping=pd.read_csv("../data/coupon_item_mapping.csv")
customer_demographics=pd.read_csv("../data/customer_demographics.csv")
customer_transaction=pd.read_csv("../data/customer_transaction_data.csv")
item=pd.read_csv("../data/item_data.csv")
test=pd.read_csv("../data/test_QyjYwdj.csv")


# In[3]:


train.head()
compaign.head()
coupon_item_mapping.head()
customer_demographics.head()
customer_transaction.head()
item.head()
test.head()


# In[4]:


#coupon_item_mapping=coupon_item_mapping.drop("coupon_index",axis=1)
#customer_transaction=customer_transaction.drop("transaction_index",axis=1)


# In[5]:


es = ft.EntitySet(id = 'train')
es= es.entity_from_dataframe(entity_id = 'train', dataframe = train, 
                              index = 'id',variable_types={"campaign_id":ft.variable_types.Index})
es=es.entity_from_dataframe(entity_id="compaign",dataframe=compaign,index='campaign_id')
es=es.entity_from_dataframe(entity_id="coupon_item_mapping",dataframe=coupon_item_mapping,index="coupon_index",make_index=True)
es=es.entity_from_dataframe(entity_id="customer_demographics",dataframe=customer_demographics,index="customer_id")
es=es.entity_from_dataframe(entity_id="customer_transaction",dataframe=customer_transaction,index="transaction_index",make_index=True)
es=es.entity_from_dataframe(entity_id="item",dataframe=item,index="item_id")

es


# ## Adding relationships between dataframes

# In[6]:


train_compaign = ft.Relationship(es['compaign']['campaign_id'],es['train']['campaign_id'])
#es.add_relationship(train_compaign)
#train_coupon=ft.Relationship(es['train']['coupon_id'],es['coupon_item_mapping']["coupon_id"])
train_cust_demog=ft.Relationship(es['customer_demographics']['customer_id'],es['train']['customer_id'])

cust_demog_trans=ft.Relationship(es['customer_demographics']['customer_id'],es['customer_transaction']['customer_id'])
cust_trans_item=ft.Relationship(es['item']['item_id'],es['customer_transaction']['item_id'])
item_coupon=ft.Relationship(es['item']['item_id'],es['coupon_item_mapping']['item_id'])


# In[7]:


es.add_relationships([train_compaign,train_cust_demog,cust_demog_trans,cust_trans_item,item_coupon])


# In[8]:


import matplotlib.pyplot as plt
#import os
#os.environ["PATH"]+='F:\anaconda\Library\bin\graphviz\dot.exe'
es.plot()


# In[15]:


#get_ipython().system('dot -V')

