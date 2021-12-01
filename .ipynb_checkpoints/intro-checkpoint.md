---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: pandoc
      format_version: 2.12
      jupytext_version: 1.11.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  nbformat: 4
  nbformat_minor: 5
---

::: {.cell .markdown}
# Introduction
:::

::: {.cell .markdown}
Recommender systems such as Spotify or Apple music are designed to offer suggestions to users for artists or songs that they may be interested in. These recommendations are based on previous listening history, or similar users listening habits. Spotify\'s recommender system uses Collaborative filtering, Natural Language Processing and audio modelling {cite}`ProducerHive`. Spotify leverages another algorithm named Bart which manages the home screen by ranking the cards and the shelves for the best engagement, while trying to provide explanations for the suggestions {cite}`mcinerney2018explore`. Netflix is another example of a widely used recommender system which presents recommendations while explaining the reason for that choice. For example, this can be due to previously watched films or popularity in that region. It does this through a variety of algorithms {cite}`gomez2015netflix`. Recommender systems aren\'t exclusive to music and streaming sites. The are also used across e-commerce, dating apps, news websites and research articles sites alike.
:::

::: {.cell .markdown}
# Approaches to Creating a Recommender System

Collaborative filtering and content based filtering are two starting points when it comes to building a recommender system {cite}`koren2009matrix`.

### Collaborative filtering

This method was first proposed in 1992 {cite}`goldberg1992using` and has become a widely used strategy for recommender systems ever since. This relies on the previous ratings of users and their similarity to other users in the past to paint a picture of their potential interests. This method doesn\'t require any domain information of the product itself as it relies on the premise that users who have similar tastes in music will be a good predictor for a unseen product based on their similarities in the past.

### Content based filtering

This method pays more attention to the product itself and can often be of use when the item has a large amount of data available which is linked to the users profile. Products can then be recommended based on previous products that the user has liked.
:::

::: {.cell .markdown}
# Our approach

This project seeks to create a music recommender system based on collaborative filtering. The system is created using code available via Google\'s introduction to recommender systems using Google Colab [*here*](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/recommendation-systems/recommendation-systems.ipynb?utm_source=ss-recommendation-systems&utm_campaign=colab-external&utm_medium=referral&utm_content=recommendation-systems#scrollTo=eSfW6SwIo4tk). It uses matrix factorisation to learn user and artist embeddings and uses stochastic gradient descent (SGD) to minimise the loss function.

### Matrix Factorisation

Matrix factorisation models map users and items to a joint latent factor space of dimensionality such that user-item interactions are modeled as inner products in that space {cite}`koren2009matrix`. In this project, matrix factorisaton characterises users and artists by vectors created using artist weighting patterns where a high correspondance will result in a recommendation.
:::

::: {.cell .markdown}
# Dataset

This dataset is from Last.FM and made available thanks to the 2nd International Workshop on Information Heterogeneity and Fusion in Recommender Systems {cite}`cantador2011proceedings`. It contains 92,800 listening records from 1,892 users across 6 files and can be accessed [*here*](https://grouplens.org/datasets/hetrec-2011/).
:::