#!/usr/bin/env python
# coding: utf-8

# # K Means Clustering with Python
# 

# #### K Means Clustering is an unsupervised learning algorithm that tries to cluster data based on their similarity.
# 
# #### Unsupervised learning means that there is no outcome to be predicted, and the algorithm just tries to find patterns in the data.
# 
# #### In k means clustering, we have to specify the number of clusters we want the data to be grouped into. 
# #### در این روش باید تعداد دسته هایی که میخواهیم داده ها در آن قرار بگیرند را مشخص کنیم
# 
# #### The algorithm randomly assigns each observation to a cluster, and finds the centroid of each cluster. 
# 
# #### Then, the algorithm iterates through two steps:
# 
# #### 1- Reassign data points to the cluster whose centroid is closest. 
# #### 2- Calculate new centroid of each cluster. 
# 
# #### These two steps are repeated till the within cluster variation cannot be reduced any further. 
# این دو مرحله تا جایی ادامه می یابد که دیگر امکان جابجایی مرکز دسته وجود نداشته باشد
# 
# #### The within cluster variation is calculated as the sum of the euclidean distance between the data points and their respective cluster centroids.
# 
# 

# # Import Libraries
# 

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Create some Data

# In[2]:


from sklearn.datasets import make_blobs


# In[3]:


# Create Data
data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)


# In[4]:


data


# In[5]:


type(data)


# In[6]:


len(data)


# In[7]:


data[0]


# In[8]:


data[1]


# In[9]:


data[0][1,0]


# In[10]:


data[0][1,1]


# In[11]:


data[0][:,0]


# In[12]:


data[0][:,1]


# # Visualize Data

# In[13]:


sns.scatterplot(x=data[0][:,0],y=data[0][:,1],hue=data[1],palette='tab10')


# In[14]:


sns.scatterplot(x=data[0][:,0],y=data[0][:,1],hue=data[1],palette='Paired')


# In[15]:


sns.scatterplot(x=data[0][:,0],y=data[0][:,1],hue=data[1],palette='rainbow')


# In[16]:


plt.scatter(x=data[0][:,0],y=data[0][:,1],c=data[1],cmap='rainbow')


# # Creating the Clusters

# In[17]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])


# In[18]:


kmeans.cluster_centers_


# In[19]:


kmeans.labels_


# In[20]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(x=data[0][:,0],y=data[0][:,1],c=kmeans.labels_,cmap='tab10')
ax2.set_title("Original")
ax2.scatter(x=data[0][:,0],y=data[0][:,1],c=data[1],cmap='tab10')


# In[23]:


kmeans = KMeans(n_clusters=3)


# In[24]:


kmeans.fit(data[0])


# In[25]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(x=data[0][:,0],y=data[0][:,1],c=kmeans.labels_,cmap='tab10')
ax2.set_title("Original")
ax2.scatter(x=data[0][:,0],y=data[0][:,1],c=data[1],cmap='tab10')


# In[ ]:




