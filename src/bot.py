import abc
import numpy as np
import praw
import time
import os
import requests
import json

from torch import cosine_similarity
from analysis import BertComparitor, PostComparisonProvider
import bot_login
import re
import pandas as pd

from db import DbService, SaveJsonToFileStrategy
from services import Service


class RedditBot: # TODO: move to a different file named bot.pys
    def __init__(self, Service: Service):
        self.reddit = Service.apiClient.getReddit()
        self.db = Service.dbService
        self.comparitor = Service.analyzerService
        self.posts = []

    def readPosts(self, subreddit = 'ProgrammingBuddies', limit = 150) -> list: # TODO: move limit and subreddit to config file
        """
        Reads posts from a subreddit and returns a list of dictionaries.
        """


        for post in self.reddit.subreddit(subreddit).new(limit = limit): 
            self.posts.append({'user': post.user, 'title': post.title,
                        'url': post.url,
                        'body' : post.selftext,
                        'score': post.score,
                        'created': post.created_utc,
                        'id': post.id})
        return self.posts[1:] # skip the first post because it's guidelines for the subreddit

    def readPostsToDataFrame(self, subreddit = 'ProgrammingBuddies', limit = 150) -> pd.DataFrame:
        """
        Reads posts from a subreddit and returns a dataframe.
        """
        columns = ['title', 'url', 'body', 'score', 'created', 'id']
        postsData = [] 
        for post in self.reddit.subreddit(subreddit).new(limit = limit):
            postData = postsData.append([ 
                        post.title,
                        post.url,
                        post.selftext,
                        post.score,
                        post.created_utc,
                        post.id])
        print('posts[1]: ', postsData[1])
        df = pd.DataFrame(np.array(postsData[1:]), # skip the first post because it's guidelines for the subreddit
         columns = columns)
        return df

    def comparePosts(self) -> any:
        """
        Compares posts and returns any.
        """
        return self.comparitor.compare(self.posts)

    
    def savePosts(self):
        """
        Saves posts to a database.
        """
        self.db.save(self.posts)

    def run(self) -> pd.DataFrame:
        """
        Runs the bot.
        """
        df = self.readPostsToDataFrame()
        return df
