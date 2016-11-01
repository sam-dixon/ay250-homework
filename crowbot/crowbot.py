import sys
import time
import datetime as dt
import ephem
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy import units as u
from keys import *
from slackclient import SlackClient

slack_client = SlackClient(CROWBOT_API)
AT_BOT = '<@{}>'.format(BOT_ID)

uh88 = ephem.Observer()
uh88.lat = '19.822991067'
uh88.lon = '-155.469433536'
uh88_loc = EarthLocation(lat=19.822991067*u.deg, lon=-155.469433536*u.deg, height=4205*u.m)
sun = ephem.Sun()
stds = []
with open('standards.txt') as f:
    while f.readline():
        l = f.readline().split()
        if len(l) > 0:
            std = {}
            std['name'] = l[0]
            std['coord'] = SkyCoord('{} {} {} {} {} {}'.format(*l[1:7]), unit=(u.hourangle, u.deg))
            std['mag'] = l[7]
            std['type'] = l[8]
            stds.append(std)


def respond(command, channel):
    """
    Handle commands.
    """
    func = not_implemented
    for k in MATCH.keys():
        if k in command:
            func = MATCH[k]
    response = func(channel)
    slack_client.api_call('chat.postMessage', as_user=True,
                          channel=channel, text=response)


def parse_slack_output(slack_rtm_output):
    """
    The Slack Real Time Messaging API is an events firehose.
    this parsing function returns None unless a message is
    directed at the Bot, based on its ID.
    """
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, whitespace removed
                return output['text'].split(AT_BOT)[1].strip().lower(), output['channel']
    return None, None


def not_implemented(channel):
    return "Sorry, I can't do that yet!"


def utc_time(channel):
    now = dt.datetime(*time.gmtime()[:6])
    return 'Current UTC time: {}'.format(now)


def sun_info(channel):
    try:
        now = dt.datetime(*time.gmtime()[:6])
        uh88.date = now
        sunset = uh88.next_setting(sun).datetime()
        uh88.horizon = '-6'
        civil = uh88.next_setting(sun).datetime()
        uh88.horizon = '-12'
        naut = uh88.next_setting(sun).datetime()
        uh88.horizon = '-18'
        astro = uh88.next_setting(sun).datetime()
        uh88.horizon = '0'
        response = ('Current UTC time: {}\n'
                    'Sunset: {}\n'
                    '6 deg. twilight: {}\n'
                    '12 deg. twilight: {}\n'
                    '18 deg. twilight: {}\n').format(now, sunset, civil, naut, astro)
    except:
        response = 'Whoops! Something went wrong calculating sun information:\n'+str(sys.exc_info())
    return response


def get_standard(channel):
    """Responds with some good standards to use"""
    now = dt.datetime(*time.gmtime()[:6])
    for std in stds:
        std['airmass'] = std['coord'].transform_to(AltAz(obstime=now, location=uh88_loc)).secz
    sorted_stds = sorted(stds, key=lambda k: abs(k['airmass']-1.0))
    first, second = sorted_stds[:2]
    return ('How about {}, a {} mag {} star at airmass {:0.4}?\n'
            'Or alternatively, {}, a {} mag {} star at airmass {:0.4}?').format(first['name'],
                                                                            first['mag'],
                                                                            first['type'],
                                                                            first['airmass'],
                                                                            second['name'],
                                                                            second['mag'],
                                                                            second['type'],
                                                                            second['airmass'])


def weather_info(channel):
    """Looks up weather info for the night"""
    return "You asked about weather. I'm working on figuring out how to respond to that!"


if __name__ == '__main__':
    READ_WEBSOCKET_DELAY = 0.5
    MATCH = {'sun': sun_info,
             'standard': get_standard,
             'std': get_standard,
             'weather': weather_info}
    if slack_client.rtm_connect():
        print("crowbot connected and running!")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                respond(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Check the Internet connect, Slack API token, and Bot ID.")
