# define scheduled jobs
import sys
from apscheduler.schedulers.background import BackgroundScheduler
from database import loadPostsAsDataFrame, upsertPostComparisons
from post_comparison import createPostsComparisons


scheduler = BackgroundScheduler()
def compute_and_update_similar_posts():
    print(f'running scheduled job: compute_and_update_similar_posts')
    # get all posts from database
    posts_df = loadPostsAsDataFrame()
    comparison_df = createPostsComparisons(posts_df)
    print('comparison_df: ', comparison_df)
    upsertPostComparisons(comparison_df)
    sys.stdout.flush()

def foo():
    print('running scheduled job: foo')
    sys.stdout.flush()

scheduler.add_job(compute_and_update_similar_posts, 'interval', hours=6, id='compare_posts')
scheduler.add_job(foo, 'interval', hours=1, id='foo')
# run the scheduler
def run_schuled_jobs():
    print('starting scheduler...')
    
    scheduler.start()