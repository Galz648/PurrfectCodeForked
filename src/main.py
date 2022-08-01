import abc
import numpy as np
import praw
import time
import os
import requests
import json

from torch import cosine_similarity

import re
import pandas as pd

from db import DbService, sqlAlchemyStrategy, db_url
from bot import RedditBot
from services import ApiClient, Service
from analysis import BertComparitor, PostComparison
        

def main(Service: Service):
    """
    Main function.
    """


    # 
    bot = RedditBot(
            Service(
                ApiClient = ApiClient(),
                db = DbService(
                        sqlAlchemyStrategy(db_url=db_url)),
                     comparitor = PostComparison(BertComparitor())
                     )
                     )

    bot.run()

    
if __name__ == "__main__":

    main(Service)
    # comparitor = BertComparitor('bert-base-nli-mean-tokens')


    # # bot setup
    # reddit = RedditBot(db = db, comparitor = comparitor)
    # reddit.run()


    # save results
    #reddit.savePosts(results)
    # print(posts)
    # reddit.savePosts(posts)



# for submission in reddit.subreddit("ProgrammingBuddies").new(limit=50):
#     print(submission)