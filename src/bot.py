import abc
import numpy as np
import praw
import time
import os
from regex import R
import requests
import json

from torch import cosine_similarity
from analysis import BertComparitor, PostComparison
import re
import pandas as pd

from db import DbService, SaveJsonToFileStrategy
from services import Service
from analysis import compare, top_n_results
from apscheduler.schedulers.blocking import BlockingScheduler



sched = BlockingScheduler()

class RedditBot:
    def __init__(self, Service: Service):
        self.reddit = Service.apiClient.getReddit()
        self.db = Service.db
        self.comparitor = Service.comparitor
        self.posts: pd.DataFrame = pd.DataFrame()

    def readPosts(self, subreddit = 'ProgrammingBuddies', limit = 150) -> pd.DataFrame:
        """
        Reads posts from a subreddit and returns a dataframe.
        """
        # filter for flair
        columns = ['author', 'title', 'url', 'body', 'score', 'created', 'id', 'flair']
        postsData = [] 
        for post in self.reddit.subreddit(subreddit).new(limit = limit):
            if post.link_flair_text == 'LOOKING FOR BUDDIES':
            # print('post author: ', post.author)
                postsData.append([ 
                            post.author,
                            post.title,
                            post.url,
                            post.selftext,
                            post.score,
                            post.created_utc,
                            post.id, 
                            post.link_flair_text])

            

        # print('posts[1]: ', postsData[1])
        return pd.DataFrame(np.array(postsData), columns = columns) # skip the first post because it's guidelines for the subreddit

    def comparePosts(self, canidate: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the comparitor to find similarities between posts.
        """
        return self.comparitor.compare(canidate, target)

    def getPersonalisedComment(self, author: str, url: str, similarity: float) -> str:
        personalisedComment = f"Hi {author}, I'm a bot. I found this post url: {url} related to yours with {similarity}% similarity.\n\n"
        return personalisedComment
    # send comments to appropriate post id
    def sendComments(self, result: pd.DataFrame):
        for index, row in result.iterrows():
            similarity = row['similarity']
            author_0 = row['author_0']
            author_1 = row['author_1']
            url_0 = row['url_0']
            url_1 = row['url_1']
            post_id_0 = row['post_id_0']
            post_id_1 = row['post_id_1']



            # send comment to first author
            comment = self.getPersonalisedComment(author_0, url_1, similarity) 
            self.postComment(post_id_0, comment, author_0)
            # send comment to second author
            comment = self.getPersonalisedComment(author_1, url_0, similarity)
            self.postComment(post_id_1, comment, author_1)

            print('comment sent')
            print('sleeping for 1 second')
            time.sleep(1)
    def postComment(self, post_id: str, comment: str, author: str) -> bool:
        # send comment to post_id
        print(f"Sending comment: {comment} to post: {post_id} with author {author}")
        submission = self.reddit.submission(post_id)
        result = submission.reply(comment)
        print('result from replying to post: ', result)
        return True

    def savePosts(self, data: pd.DataFrame):
        """
        Saves posts to a database.
        """
        self.db.save(data)

    @sched.scheduled_job('interval', seconds=5)
    def timed_job():
        print('This job is run every five seconds.')

    def loadPosts(self):
        """
        Loads posts from a database.
        """
        return self.db.load()

    def start(self):
        pass
    def run(self) -> pd.DataFrame:
        """
        Runs the bot.
        gets the saved posts from the database and compares them with the new posts
        """
       # get all the posts
        posts_df = self.loadPosts()
        print('loaded posts: ', posts_df)
        # get all the posts
        if  posts_df is None or posts_df.empty:
            print('no posts found locally')
            print('reading posts from reddit')
            posts_df = self.readPosts()

            print('saving posts to database')
            self.savePosts(posts_df)
            print('posts saved to database')

        # get new posts from reddit
        print('reading new posts from reddit')
        new_posts_df = self.readPosts(limit=5)
        print('new posts read from reddit')
        # compare new posts with old posts
        print('comparing new posts with old posts')
        result_df = self.comparePosts(new_posts_df, posts_df)
        print('comparison result df: ', result_df)
        # run the scheduler every hour
        # print('starting scheduler...')
        # sched.start()







        # compare the posts to themselves
        # df, df_so = compare(df, "") # move this function to the compare class
        # # get top 100 results
        # result = top_n_results(df, df_so) # move this function to the compare class
        # print('top n results:', result)
        # save results to database

        # df = self.loadPosts()
        # print('loaded posts: ', df)




