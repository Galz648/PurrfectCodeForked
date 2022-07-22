import praw
import os
from dotenv import load_dotenv
from analysis import PostComparisonProvider

from db import DbService

load_dotenv()


    

    

class ApiClient:
    def __init__(self):
        self.api_config = {
            'reddit_username': os.environ.get("reddit_username"),
            'reddit_password': os.environ["reddit_password"],
            "client_id": os.environ["client_id"],
            "client_secret": os.environ["client_secret"],
            "user_agent": "reddit_bot_v1.0"
        }
    

    def getReddit(self) -> praw.Reddit:
        return praw.Reddit(**self.api_config)


class Service:
    def __init__(self, ApiClient: ApiClient, DbService: DbService, analyzerService: PostComparisonProvider) -> None:
        self.apiClient = ApiClient
        self.dbService = DbService
        self.analyzerService = analyzerService
    # # db setup
    # db_strategy = SaveJsonToFileStrategy
    # db = DbProvider(db_strategy())
