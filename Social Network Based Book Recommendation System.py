#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx
from operator import itemgetter
import matplotlib.pyplot
import pandas as pd


# In[2]:


# Read the data from amazon-books.csv into amazonBooks dataframe;
amazonBooks = pd.read_csv('/Users/bingyandu/amazon1-books.csv', index_col=0)


# In[3]:


# Read the data from amazon-books-copurchase.adjlist;
# assign it to copurchaseGraph weighted Graph;
# node = ASIN, edge= copurchase, edge weight = category similarity
fhr=open("amazon-books-copurchase.edgelist", 'rb')
copurchaseGraph=networkx.read_weighted_edgelist(fhr)
fhr.close()


# In[4]:


# Now let's assume a person is considering buying the following book;
# what else can we recommend to them based on copurchase behavior 
# we've seen from other users?
print ("Looking for Recommendations for Customer Purchasing this Book:")
print ("--------------------------------------------------------------")
purchasedAsin = '0805047905'


# In[5]:


# Let's first get some metadata associated with this book
print ("ASIN = ", purchasedAsin) 
print ("Title = ", amazonBooks.loc[purchasedAsin,'Title'])
print ("SalesRank = ", amazonBooks.loc[purchasedAsin,'SalesRank'])
print ("TotalReviews = ", amazonBooks.loc[purchasedAsin,'TotalReviews'])
print ("AvgRating = ", amazonBooks.loc[purchasedAsin,'AvgRating'])
print ("DegreeCentrality = ", amazonBooks.loc[purchasedAsin,'DegreeCentrality'])
print ("ClusteringCoeff = ", amazonBooks.loc[purchasedAsin,'ClusteringCoeff'])


# In[6]:


# Now let's look at the ego network associated with purchasedAsin in the
# copurchaseGraph - which is esentially comprised of all the books 
# that have been copurchased with this book in the past
# (1) YOUR CODE HERE: 
#     Get the depth-1 ego network of purchasedAsin from copurchaseGraph,
#     and assign the resulting graph to purchasedAsinEgoGraph.
dep1ego = networkx.ego_graph(copurchaseGraph,purchasedAsin,radius=1)
purchasedAsinEgoGraph = networkx.Graph(dep1ego)


# In[7]:


# Next, recall that the edge weights in the copurchaseGraph is a measure of
# the similarity between the books connected by the edge. So we can use the 
# island method to only retain those books that are highly simialr to the 
# purchasedAsin
# (2) YOUR CODE HERE: 
#     Use the island method on purchasedAsinEgoGraph to only retain edges with 
#     threshold >= 0.5, and assign resulting graph to purchasedAsinEgoTrimGraph
threshold = 0.5
purchasedAsinEgoTrimGraph = networkx.Graph()
for i,n,v in purchasedAsinEgoGraph.edges(data=True):
    if v['weight'] >= threshold:
        purchasedAsinEgoTrimGraph.add_edge(i,n,weight=v['weight'])


# In[8]:


# Next, recall that given the purchasedAsinEgoTrimGraph you constructed above, 
# you can get at the list of nodes connected to the purchasedAsin by a single 
# hop (called the neighbors of the purchasedAsin) 
# (3) YOUR CODE HERE: 
#     Find the list of neighbors of the purchasedAsin in the 
#     purchasedAsinEgoTrimGraph, and assign it to purchasedAsinNeighbors
purchasedAsinNeighbors = purchasedAsinEgoTrimGraph.neighbors(purchasedAsin)
print(list(purchasedAsinEgoTrimGraph.neighbors(purchasedAsin)))


# In[9]:


# Next, let's pick the Top Five book recommendations from among the 
# purchasedAsinNeighbors based on one or more of the following data of the 
# neighboring nodes: SalesRank, AvgRating, TotalReviews, DegreeCentrality, 
# and ClusteringCoeff
# (4) YOUR CODE HERE: 
#     Note that, given an asin, you can get at the metadata associated with  
#     it using amazonBooks (similar to lines 29-36 above).
#     Now, come up with a composite measure to make Top Five book 
#     recommendations based on one or more of the following metrics associated 
#     with nodes in purchasedAsinNeighbors: SalesRank, AvgRating, 
#     TotalReviews, DegreeCentrality, and ClusteringCoeff. Feel free to compute
#     and include other measures if you like.
#     YOU MUST come up with a composite measure.
#     DO NOT simply make recommendations based on sorting!!!
#     Also, remember to transform the data appropriately using 
#     sklearn preprocessing so the composite measure isn't overwhelmed 
#     by measures which are on a higher scale.

#plug in the metadata for purchasedAsinNeighbors
asinmeta = []
for n in purchasedAsinNeighbors:
    ASIN = n
    Title = amazonBooks.loc[n,'Title']
    SalesRank = amazonBooks.loc[n,'SalesRank']
    TotalReviews = amazonBooks.loc[n,'TotalReviews']
    AvgRating = amazonBooks.loc[n,'AvgRating']
    DegreeCentrality = amazonBooks.loc[n,'DegreeCentrality']
    ClusteringCoeff = amazonBooks.loc[n,'ClusteringCoeff']
    asinmeta.append((ASIN, Title,SalesRank,TotalReviews,AvgRating,DegreeCentrality,ClusteringCoeff))

asinmeta 


# In[10]:


#transform the metadata into DataFrame, using ASIN as the index
asinmeta_df = pd.DataFrame(data = asinmeta,columns=['ASIN','Title','SalesRank',
                                                    'TotalReviews','AvgRating',
                                                   'DegreeCentrality',
                                                    'ClusteringCoeff'])
asinmeta_df.set_index('ASIN', inplace=True)


# In[11]:


#use mix-max-scale to standardize the AvgRating column
from sklearn import preprocessing
x = asinmeta_df[['AvgRating']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
std_AvgRating = pd.DataFrame(x_scaled,index=asinmeta_df.index,columns=['std_AvgRating'])
std_AvgRating.shape


# In[12]:


#use mix-max-scale to standardize the TotalReviews column
y = asinmeta_df[['TotalReviews']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y)
std_TotalReviews = pd.DataFrame(y_scaled,index=asinmeta_df.index,columns=['std_TotalReviews'])


# In[13]:


#concat the standardized TotalReviews to the original dataframe
asinmeta_df=pd.concat([asinmeta_df,std_TotalReviews],axis=1)
asinmeta_df.head()


# In[14]:


#concat the standardized AvgRating to the original dataframe
asinmeta_df=pd.concat([asinmeta_df,std_AvgRating],axis=1)
asinmeta_df.head()


# In[15]:


asinmeta_df=asinmeta_df.copy()


# In[16]:


#Since The lower the rank, the better a product is selling. I used inverse min-max-scaling for the Sales Rank
z= asinmeta_df['SalesRank']
new_salesrank = []
for i in z:
    i=(max(z)-i)/(max(z)-min(z))
    new_salesrank.append(i)

std_SalesRank = pd.DataFrame(new_salesrank,index=asinmeta_df.index,columns=['std_SalesRank'])


# In[17]:


#concat standardized SalesRank to the original dataframe
asinmeta_df=pd.concat([asinmeta_df,std_SalesRank],axis=1)
asinmeta_df.head()


# In[18]:


#Method 1: please refer to the descrption for the details
a=asinmeta_df['std_TotalReviews']+asinmeta_df['std_AvgRating']+asinmeta_df['std_SalesRank']
asinmeta_df['Composite_Mesuare']=a*asinmeta_df['DegreeCentrality']
asinmeta_df


# In[19]:


#Method 2: please refer to the descrption for the details
asinmeta_df['Composite_Mesuare1']=a*asinmeta_df['ClusteringCoeff']


# In[20]:


# Using sorting method by the value of method 1 result, select top 5 and assign to a new object.
asinmeta_final=asinmeta_df.sort_values(by=['Composite_Mesuare'],ascending=False)[:5]
asinmeta_final.drop(['std_TotalReviews','std_AvgRating','std_SalesRank','Composite_Mesuare','Composite_Mesuare1'],axis=1)


# In[21]:


# Using sorting method by the value of method 2 result, select top 5 and assign to a new object.
asinmeta_final1=asinmeta_df.sort_values(by=['Composite_Mesuare1'],ascending=False)[:5]


# In[22]:


# Print Top 5 recommendations (ASIN, and associated Title, Sales Rank, 
# TotalReviews, AvgRating, DegreeCentrality, ClusteringCoeff)
# (5) YOUR CODE HERE:  

#print top 5 recommendations for Method1
print()
print('Top 5 Recommendation for',amazonBooks.loc[purchasedAsin]['Title'],'is:')

print(asinmeta_final)


# In[23]:


#print top 5 recommendations for Method2
print()
print('Top 5 Recommendation for',amazonBooks.loc[purchasedAsin]['Title'],'is:')

print(asinmeta_final1)


# In[ ]:




