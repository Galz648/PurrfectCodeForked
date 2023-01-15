# define scheduled jobs
from apscheduler.schedulers.background import BackgroundScheduler
from database import loadPostsAsDataFrame, upsertPostComparisons
from post_comparison import createPostsComparisons


scheduler = BackgroundScheduler(daemon=True)
def compute_and_update_similar_posts():
    print(f'running scheduled job: compute_and_update_similar_posts')
    # get all posts from database
    posts_df = loadPostsAsDataFrame()
    comparison_df = createPostsComparisons(posts_df)
    print('comparison_df: ', comparison_df)
    upsertPostComparisons(comparison_df)

def foo():
    print('running foo')


scheduler.add_job(compute_and_update_similar_posts, 'interval', hours=4, id='compare_posts')
# scheduler.add_job(foo, 'interval', seconds=5, id='foo')
# run the scheduler
def run_schuled_jobs():
    print('starting scheduler...')
    
    scheduler.start()