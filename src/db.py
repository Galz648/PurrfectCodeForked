
import abc
import pandas as pd
import uuid
from time import time

class DatabaseClientStrategy(abc.ABC):
    
    @abc.abstractmethod
    def save(self, data: pd.DataFrame) -> None:
        pass
    
    def load(self) -> pd.DataFrame:
        pass


from reddit_types import DataType
from sqlalchemy import ForeignKey, MetaData, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean

Base = declarative_base()
db_url = 'sqlite:///lightdb.db'



class Post(Base):
    __tablename__ = 'posts'

    id = Column(Integer, primary_key=True)
    author = Column(String)
    post_id = Column(String)
    post_title = Column(String)
    post_body = Column(String)
    post_time = Column(Float)
    post_url = Column(String)
    post_flair = Column(String)
    post_score = Column(Integer)
    post_clean_text = Column(String)

class PostMatch(Base):
    __tablename__ = 'post_matches'

    id = Column(Integer, primary_key=True)
    match_time = Column(Float)

    author = Column(String)
    comparison_author = Column(String)
    post_id = Column(Integer, ForeignKey("posts.id")) # foreign key to posts table
    match_id = Column(String)
    match_post_id =  Column(Integer, ForeignKey("posts.id")) # foreign key to posts table
    match_post_time = Column(Float)
    similarity_score = Column(Float)
class sqlAlchemyStrategy(DatabaseClientStrategy):
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(self.db_url, echo=True)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.con = self.engine.connect()
        Base.metadata.create_all(self.engine)

    def save_post(self, post: Post) -> None:
        self.save(post)

    def close(self) -> None:
        self.session.close()
        self.engine.dispose()

    def save(self, data: pd.DataFrame) -> None:
        for index, row in data.iterrows():
                post = Post(
                    author=str(row['author']),
                    post_id=row['id'],
                    post_title=row['title'],
                    post_body=row['body'],
                    post_time=row['created'],
                    post_url=row['url'],
                    post_flair=row['flair'],
                    post_score=row['score'],
                    post_clean_text = row['text']
                )
                self.session.add(post)

        self.session.commit()
        self.close()

    def load(self) -> pd.DataFrame:
        self.con = self.engine.connect()
        # read posts from 'posts' table in database 
        posts = pd.read_sql_table('posts', self.con)
        print('posts: ', posts)
        return posts
class SaveJsonToFileStrategy(DatabaseClientStrategy):
    def save(self, data: pd.DataFrame, type: DataType, file_format: str = "json") -> None:
        filename = f"{DataType}{time.now()}.{file_format}"
        data.json(filename, index=False)

    def load(self) -> pd.DataFrame:
        raise NotImplementedError


class DbService:
    """
    This class provides abstraction from the database
    """

    def __init__(self, strategy: DatabaseClientStrategy):
        self.strategy = strategy

    def save(self, data: pd.DataFrame) -> None:
        self.strategy.save(data)

    def load(self) -> pd.DataFrame:
        return self.strategy.load()

