"""
Poking around at fasola data.
"""

import imageio
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess as sp
import matplotlib.pyplot as plt
from ast import literal_eval

sns.set_style('white')

#sns.palplot(sns.color_palette("GnBu_d"))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)

base_data_path = '/Users/jonas/Desktop/Programming/Python/fasola/data/'
dir_path = '/Users/jonas/Desktop/Programming/Python/fasola/figures/'

# Get years:
years = range(2018, 1994, -1)

# Find the least popular song (for scaling y-axes)
max_rank = len(songs_df['page'])


# Load in the data
def load_data_csv(metric='rank'):
    """I think this is garbage. Just use fucking pickles."""
    if metric == 'rank':
        path = base_data_path + 'song_data_rank.csv'
    elif metric == 'count':
        path = base_data_path + 'song_data_count.csv'
    else:
        return "Please choose 'rank' or 'count' for metric."

    songs_df = pd.read_csv(path, sep=',') #, dtype={'page': str, 'ranks': np.array})

    literal_eval(songs_df['ranks'][0])
    # Reading in the CSV pulls the ranks in as a string, so fix that to a list.
    for rs in range(len(songs_df['ranks'])):
        songs_df['ranks'][rs] = literal_eval(songs_df['ranks'][rs])
    return songs_df

def load_data_pickle(metric='rank'):
    if metric == 'rank':
        path = base_data_path + 'song_data_rank.pickle'
    elif metric == 'count':
        path = base_data_path + 'song_data_count.pickle'
    else:
        return "Please choose 'rank' or 'count' for metric."

    songs_df = pd.read_pickle(path)
    return songs_df


songs_df = load_data_pickle(metric='rank')



# Get pages. This is pretty unnecessary but just for kicks.
pages = []
for p in songs_df['page']:
    try:
        pages.append(int(p))
    except ValueError:
        pass
max(pages)



# Some generally useful functions
def get_page_idx(page):
    """Find the index of a song."""
    try:
        idx = songs_df.where(songs_df['page'] == page)
        page_idx = idx.dropna()['page'].index[0]
        return page_idx

    except IndexError:
        return "Invalid page number (try {}t or {}b)".format(page, page)

def get_song_rank(page, year=2018):
    year_idx = max(years) - year
    page_idx = get_page_idx(page)
    rank = songs_df['ranks'][page_idx][year_idx]
    return rank




# Do some plotting
def plot_one_song(page, save=True):
    """Plot the evolution and overall rank of a given song."""
    plt.close()

    page_idx = get_page_idx(page)
    if type(page_idx) == str:
        return page_idx

    plt.plot(years, songs_df['ranks'][page_idx], '-b',
             linestyle='-', color='b', linewidth='3',
             label='Annual Rank')
    plt.plot(years, songs_df['ranks'][page_idx], 'ob')

    plt.plot(years, [songs_df['ranks'][page_idx][0]] * len(years),
             linestyle='--', color='r',
             label='Overall Rank')
    # plt.gca().invert_yaxis()
    plt.ylim(max(pages), -20)
    plt.ylabel('Rank', weight='bold')
    plt.xlabel('Year', weight='bold')
    plt.yticks([1, 250, max_rank])
    plt.title('Song Call Ranking for ' + page, weight='bold')
    plt.legend()

    if save:
        plt.savefig(dir_path + 'song-popularity_' + page + '.png')
    plt.show()

plot_one_song('344')


def plot_songs_popularity(ps, save=True):
    """
    Plot the evolution and overall rank of a set of songs.

    This is basically the same function as the one-song, just with some
    switched-up labeling and so on.
    """
    plt.close()

    # Give the user a little flexibility on input types.
    if type(ps) == str or type(ps) == int:
        ps = [str(ps)]

    colors = sns.diverging_palette(220, 20, n=len(ps), center='dark')

    if len(ps) > 7:
        print "Plotting under protest: probably too many songs to look good."

    for p in range(len(ps)):
        page = ps[p]
        page_idx = get_page_idx(page)
        if type(page_idx) == str:
            return page_idx

        plt.plot(years, songs_df['ranks'][page_idx],
                 linestyle='-', linewidth='3', alpha=0.7,
                 color=colors[p], label=page)
        plt.plot(years, [songs_df['ranks'][page_idx][0]] * len(years),
                 linestyle='--', color=colors[p], alpha=0.4)

    # plt.gca().invert_yaxis()
    plt.ylim(max(pages), -20)
    plt.ylabel('Rank', weight='bold')
    plt.xlabel('Year', weight='bold')
    plt.yticks([1, 250, max_rank])

    plt.title('Song Call Ranking for Songs ' + ', '.join(ps), weight='bold')
    plt.legend()

    if save:
        plt.savefig(dir_path + 'song-popularity_' + '-'.join(ps) + '.png')
    plt.show()

plot_songs_popularity(ex_songs)

plot_songs_popularity(344)



# Make a gif of some songs evolutions.
ex_songs = ['178', '24t', '344', '245']
def gif_evo(songs=ex_songs, show_plots=True, dt=0.4):
    base_fname = 'songs-evo_' + '-'.join(songs)
    gif_outpath = dir_path + '/' + base_fname + '.gif'

    sp.call(['rm', '-f', '{}'.format(gif_outpath)])

    # Set a color palette
    sns.diverging_palette(220, 20, n=len(songs), center='dark')

    images = []
    for year in years[::-1]:
        ranks = [max_rank - get_song_rank(song, year) for song in ex_songs]

        sns.barplot(x=ex_songs, y=ranks, palette=sns.color_palette("GnBu_d"))

        plt.ylim(600, -20)
        plt.ylim(-20, 557)
        # plt.yticks([-20, 250, 500], [500, 250, 1])
        plt.yticks([1, 250, max_rank], [max_rank, 250, 1])
        # plt.yticks([0, max_rank], ['Least Popular', 'Most Popular'])
        plt.ylabel('Rank', weight='bold')
        plt.xlabel('Song', weight='bold')
        plt.title('Song Call Rankings in ' + str(year), weight='bold')

        file_name = dir_path + base_fname + '_{}.png'.format(str(year))
        plt.savefig(file_name)

        if show_plots:
            plt.show()
            plt.close()

        images.append(imageio.imread(file_name))

    imageio.mimsave(gif_outpath, images, duration=dt)

    for year in years:
        file_name = dir_path + base_fname + '_{}.png'.format(str(year))
        sp.call(['rm', '-rf', '{}'.format(file_name)])

gif_evo(show_plots=True)





# Analysis stuff
"""
Some analysis ideas:
    - Find which songs have the most change
"""







# The End
