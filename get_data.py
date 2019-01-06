"""
Scrape the data off fasola.org.
"""

import pickle
import requests
import pandas as pd
from bs4 import BeautifulSoup

# bs4_object.prettify() is a thing
url_rank = 'https://fasola.org/minutes/stats/?c=2018&fbclid=IwAR34jm_keTkqaMjQos2VUKWc8GOj3GLap5bRzQ3IUhM-rJkpK9zuXIdqQyg'
url_count = 'https://fasola.org/minutes/stats/?c=2018&s=c'



def get_data(url, filename='song_data_rank', outtype='pickle'):
    def get_url(url):
        """Get a body."""
        response = requests.get(url)
        html = response.content
        soup = BeautifulSoup(html, 'lxml')
        body = soup.find('body')
        return body

    body = get_url(url)
    container = body.find(attrs={'class': 'MinStatTable'})
    container_list = container.prettify().split('</tr>')


    # Get the years and headers
    years = range(2018, 1994, -1)

    # Get the songs
    songs = []
    for k in range(1, len(container_list) - 1):
        song = container_list[k].split()
        song = [str(element) for element in song]

        this_song = []
        for i in range(len(song)):
            if song[i][0] != '<' and song[i][0] != 'h':
                this_song.append(song[i])
        song_dict = {'page': this_song[0],
                     'ranks': [int(rank) for rank in this_song[2:]],
                     'overall_rank': int(this_song[1])}
        songs.append(song_dict)
    songs_df = pd.DataFrame(songs)

    if outtype == 'csv':
        songs_df.to_csv('./{}.csv'.format(filename), sep=',', index=False)
    elif outtype == 'pickle':
        songs_df.to_pickle('./{}.pickle'.format(filename))
    else:
        return "Please choose 'csv' or 'pickle' for outtype."

get_data(url_rank, filename='song_data_rank')
get_data(url_count, filename='song_data_count')
