import time
import schedule
import tweepy
from argparse import ArgumentParser
from src.generator import Generator
import logging
from logging import getLogger
from src.log import init_logging
init_logging(level = logging.INFO)
logger = getLogger(__name__)

def gen_api(CK, CS, AK, AS):
    auth = tweepy.OAuthHandler(CK, CS)
    auth.set_access_token(AK, AS)
    api = tweepy.API(auth)
    return api


def jan_lisa(api, gen):
    user = 'LeeKaixin2003'
    tweet_id = 1394936576078090244
    try:
        x = gen.generate()
        x = '@{} {}'.format(user, x)
        api.update_status(
                status = x,
                in_reply_to_status_id = tweet_id)
        logger.info('tweeted: {}'.format(x))
    except Exception as e:
        logger.info('failed:', e)


def main():
    parser = ArgumentParser()
    parser.add_argument('--consumer-key')
    parser.add_argument('--consumer-secret')
    parser.add_argument('--api-key')
    parser.add_argument('--api-secret')
    args = parser.parse_args()

    api = gen_api(
            CK = args.consumer_key,
            CS = args.consumer_secret,
            AK = args.api_key,
            AS = args.api_secret)
    gen = Generator()

    schedule.every().minute.at(':00').do(lambda : jan_lisa(api, gen))
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    main()

