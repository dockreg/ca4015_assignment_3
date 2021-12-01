#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# We will explore the 6 files of the lastfm data set to further understand the data within and to investigate relationships between the users, artists and tags before we proceed with our recommender system.

# ## Install dependencies

# In[1]:


from __future__ import print_function
import seaborn as sns
import numpy as np
import pandas as pd
import collections
from mpl_toolkits.mplot3d import Axes3D
from IPython import display
from matplotlib import pyplot as plt
import sklearn
import sklearn.manifold
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


# ## Import data
# 
# Import data from our 6 separate files and do some preliminary analysis to better understand what information is contianed in this dataset.
# 
# We have 6 files:
# - artists
# - tags
# - user_artists
# - user_friends
# - user_taggedartists-timestamp
# - user_taggedartists
# 

# ### Artists

# In[2]:


# Load artists
artists_cols = ['id', 'name', 'url', 'pictureURL']
artists = pd.read_csv('../Data/artists.dat', sep='	', names=artists_cols, skiprows=1)
artists.head(5)


# In[3]:


artists.info()


# The picture URL contains some null values but we will not need this column for our analysis and so we can drop this column and the url column.

# In[4]:


artists.drop('pictureURL', axis=1, inplace=True)
artists.drop('url', axis=1, inplace=True)


# In[5]:


artists.head()


# This dataframe contains one row for each of the 17,632 artist in this data set with their corresponding id

# ### Tags

# In[6]:


# Load tags
tags_cols = ['tagID', 'tagValue']
tags = pd.read_csv('../Data/tags.dat', sep='	', encoding='latin-1')
tags.head()


# In[7]:


tags.info()


# In[8]:


tags.describe()


# In[9]:


tags1 = tags[tags['tagValue'].str.endswith('metal')]
tags1.value_counts()


# This dataframe contains one row for each of the 11,946 seperate tags that can be applied to each artist. As we can see from above there can be a wide variety of different genres. There are 306 different tags that all contain the word metal. We will need to be minfuk of this when doing analysis.

# ### User Artists

# In[10]:


# Load user-artists
user_artists_cols = ['userID', 'artistID', 'weight']
user_artists = pd.read_csv('../Data/user_artists.dat', sep='	')
user_artists.head()


# In[11]:


user_artists.describe()


# In[12]:


user_artists.value_counts()


# The user artists contains users, the artist they listen to, and the weight which is proportional to how much they have listened to the artist. The weight value goes from 1-352,698 with an average weight of 745. Users may have a weighting for multiple artists.
# 
# (README.txt)
# 92834 user-listened artist relations:
#          avg. 49.067 artists most listened by each user
#          avg. 5.265 users who listened each artist

# ### User Friends

# In[13]:


# Load user-friends
user_friends_cols = ['userID', 'friendID']
user_friends = pd.read_csv('../Data/user_friends.dat', sep='	')
user_friends.head()


# In[14]:


user_friends.describe()


# This dataframe contains: 12717 bi-directional user friend relations, i.e. 25434 (user_i, user_j) pairs
#          avg. 13.443 friend relations per user (taken from README.txt with the data)
#          
# We will explore this data to see if any clear patterns emerge and to see what the distribution of friendship is like in this data.

# In[15]:


top_friends = user_friends[['userID', 'friendID']].groupby('userID').count().reset_index()
top_friends.rename({'friendID':'count'}, axis=1, inplace=True)

top_friends = top_friends.sort_values('count', ascending=False)
top_friends


# In[16]:


r = sns.color_palette('Paired')
sns.boxplot(x=top_friends['count'], palette=r)

plt.title('Top Friends')
plt.xlabel('Count of friends')
plt.show()


# This boxplot gives us an idea of the relationships between friends. The majority of the users have < 20 friends however there appear to be some outliers who have upwards of 40 and as many as 119 friends. These are very influential users within this dataset

# ### User Tagged Artists Timestamp

# In[17]:


# Load user-tagged-artists-timestamps
user_tagged_artists_tstamp_cols = ['userID', 'artistID', 'tagID', 'timestamp']
user_tagged_artists_tstamp = pd.read_csv('../Data/user_taggedartists-timestamps.dat', sep='	')
user_tagged_artists_tstamp.head()


# In[18]:


user_tagged_artists_tstamp.describe()


# ### User Tagged Artists

# In[19]:


# Load user-tagged-artists
user_tagged_artists_cols = ['userID', 'artistID', 'tagID', 'day', 'month', 'year']
user_tagged_artists = pd.read_csv('../Data/user_taggedartists.dat', sep='	')
user_tagged_artists.head()


# In[20]:


user_tagged_artists.describe()


# The users tagged artist and users tagged artists timestamp are the same data across userID, artistID and tagID columns.

# ## Data Manipulation and Visualisation
# 
# We will bring all of our data together and create some visualisations to better understand the data.
# 

# In[21]:


artists.tail()


# It is noted the index and the id don't add up at the end of the dataframe so we need to be aware of this when we come to the recommender system.

# In[22]:


user_artists.tail()


# In[23]:


tags.tail()


# In[24]:


user_tagged_artists.tail()


# In[25]:


user_friends.tail()


# # Merge Data / Visualisations
# 

# In[26]:


merged_uta_t = pd.merge(user_tagged_artists, tags, on = 'tagID')


# In[27]:


merged_uta_t.head()


# We want to create a calculated field counting how many of each unique tag have been applied. This will show us the most listened to genres. 

# In[28]:


top_tag = merged_uta_t[['userID', 'tagValue']].groupby('tagValue').count().reset_index()
top_tag.rename({'userID':'count'}, axis=1, inplace=True)

#limit top tags to top 10
top_tag = top_tag.sort_values('count', ascending=False).head(10)
top_tag


# In[29]:


#Display top ten tags
r = sns.color_palette('Paired')
ax = sns.barplot(x='count', y='tagValue', data=top_tag,
            label="tagValue", palette=r)

plt.title('Top Tags per genre')
plt.ylabel('Tag')
plt.xlabel('Count')
plt.show()


# We can see rock is the dominant genre. Rock is the number one choice, but as we alluded to earlier, there is also alternative rock and classic rock as the 9th and 10th most popular choices.

# In[30]:


merged_ua_a = pd.merge(user_artists, artists, how='left', left_on='artistID', right_on='id')
merged_ua_a.head()


# We now look at the different weights given to each of the artists indicating how much each artist has been listened to by users.

# In[31]:


top_artist = merged_ua_a[['weight', 'name']].groupby('name').sum().reset_index()

top_artist


# In[32]:


#Display distribution of artists cumulative weights
sns.histplot(x='weight', data=top_artist, palette=r, bins=20)
plt.title('Distribution of weight across Artists')
plt.ylabel('Count')
plt.xlabel('Weight')
plt.show()


# In[33]:


top_artist.describe()


# This data is extremely right skewed with some large outliers (max value of 2,393,140) which is making it difficult to visualise the distribution. We will use log scale to enable us to do so.

# In[34]:


#Display artists weights- log
sns.histplot(x='weight', data=top_artist, palette=r, log_scale=True, bins=20)
plt.title('Distribution of weight across Artists - LOG SCALE')
plt.ylabel('Count')
plt.xlabel('Weight')
plt.show()


# There appears to be a normal distribution around the log scale of weights assigned to each artist.
# 
# ## Conclusion 
# 
# We have completed our preliminary analysis of the data provided. We understand the weights value varies greatly across the data set. We also have an indication that there are a large number of highly linked users and friends across the data. We are hopeful that with all of this data we will be able to create a useful music recommender system.

# In[ ]:




