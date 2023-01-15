import sys
from reddit import subreddit
from scheduled_jobs import run_schuled_jobs
from database import is_looking_for_buddies_flare, post_exists, insert_post, create_post_info



def run_posts_stream_processing():
    print('running stream processing')
    for submission in subreddit.stream.submissions():
        if is_looking_for_buddies_flare(submission):
            print(f'post: {submission} \n')

            if not post_exists(submission.id):
                post_info = create_post_info(submission)
                is_created  = insert_post(post_info)
                print('post created: ', is_created)

            else:
                print('post already exists')

        sys.stdout.flush()

def main():
    run_schuled_jobs()
    run_posts_stream_processing()

if __name__ == "__main__":
    main()

