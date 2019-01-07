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
cmap = sns.cubehelix_palette(light=1, as_cmap=True)

base_data_path = '/Users/jonas/Desktop/Programming/Python/fasola/data/'
fig_path = '/Users/jonas/Desktop/Programming/Python/fasola/figures/'

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


# Some plotters
def plot_songs_popularity(ps, save=True, show=True, fig=None, ax=None):
    """
    Plot the evolution and overall rank of a set of songs.

    This is basically the same function as the one-song, just with some
    switched-up labeling and so on.
    """
    plt.close()

    # If no external axes are provided, set one up.
    if ax == None:
        fig, ax = plt.subplots()
    # If external axes are provided, make sure we don't save or show here.
    else:
        show, save = False, False

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

        ax.plot(years, songs_df['ranks'][page_idx],
                 linestyle='-', linewidth='3', alpha=0.7,
                 color=colors[p], label=page)
        ax.plot(years, [songs_df['ranks'][page_idx][0]] * len(years),
                 linestyle='--', color=colors[p], alpha=0.4)

    # plt.gca().invert_yaxis()
    ax.set_ylim(max(pages), -20)
    ax.set_ylabel('Rank', weight='bold')
    ax.set_xlabel('Year', weight='bold')
    ax.set_yticks([1, 250, max_rank])
    sns.despine()

    songs = '' if len(ps) == 1 else 'Songs '
    ax.set_title('Song Call Ranking for ' + songs + ', '.join(ps), weight='bold')
    ax.legend()

    if save:
        plt.savefig(fig_path + 'song-popularity_' + '-'.join(ps) + '.png')
    if show:
        plt.show()

plot_songs_popularity(ex_songs)

plot_songs_popularity(344)



# Make a gif of some songs evolutions.
ex_songs = ['178', '24t', '344', '245']
def evo_gif(songs=ex_songs, show_plots=True, dt=0.4):
    base_fname = 'songs-evo_' + '-'.join(songs)
    gif_outpath = fig_path + '/' + base_fname + '.gif'

    sp.call(['rm', '-f', '{}'.format(gif_outpath)])

    # Set a color palette
    sns.diverging_palette(220, 20, n=len(songs), center='dark')

    images = []
    for year in years[::-1]:
        ranks = [max_rank - get_song_rank(song, year) for song in ex_songs]

        sns.barplot(x=ex_songs, y=ranks,
                    palette=sns.diverging_palette(220, 20, n=len(songs),
                                                  center='dark'))

        plt.ylim(600, -20)
        plt.ylim(-20, 557)
        # plt.yticks([-20, 250, 500], [500, 250, 1])
        plt.yticks([1, 250, max_rank], [max_rank, 250, 1])
        # plt.yticks([0, max_rank], ['Least Popular', 'Most Popular'])
        plt.ylabel('Rank', weight='bold')
        plt.xlabel('Song', weight='bold')
        plt.title('Song Call Rankings in ' + str(year), weight='bold')
        sns.despine()

        file_name = fig_path + base_fname + '_{}.png'.format(str(year))
        plt.savefig(file_name)

        if show_plots:
            plt.show()
            plt.close()

        images.append(imageio.imread(file_name))

    imageio.mimsave(gif_outpath, images, duration=dt)

    for year in years:
        file_name = fig_path + base_fname + '_{}.png'.format(str(year))
        sp.call(['rm', '-rf', '{}'.format(file_name)])

evo_gif(show_plots=True)




songs_df['page']
# Analysis stuff
"""
Some analysis ideas:
    - Find which songs have the most change
"""

def get_var(df=songs_df):
    """
    A little handmade variance calculator.
    Needless to say, the numpy one is way better, so this is useless.
    """
    n_years = len(years)
    n_songs = len(songs_df['page'])
    variances = np.zeros((n_songs, n_songs))
    annual_diffs = np.zeros((n_songs, n_songs, n_years))

    # Figure out how to just get upper/lower triangle rather than populating w dups
    for s1 in range(n_songs):
        for s2 in range(n_songs):
            s1_ranks = songs_df['ranks'][s1]
            s2_ranks = songs_df['ranks'][s2]

            # Set up an offset/normalizer so that we're just looking at
            # functional form, not call totals. Maybe do this as a frac instead.
            offset = s1_ranks[0] - s2_ranks[0]

            annual_difference = [s1_ranks[year] - s2_ranks[year] - offset for year in range(n_years)]
            variance = sum( (annual_difference - np.mean(annual_difference))**2)/float(n_years)

            variances[s1][s2] = variance
            annual_diffs[s1][s2] = annual_difference


    mask = np.zeros_like(variances)
    mask[np.triu_indices_from(mask)] = True
    corr_matrix=variances.corr()

    sns.heatmap(variances, mask=mask) #, vmin=510, vmax=530)
    plt.show()
    return variances
vars = get_var()


def get_covariances(songs=songs_df, method='corrcoef', save=False, out_as='png'):
    """ Make a covariance matrix."""
    # Make a matrix out of the rankings.
    songs_mat = np.matrix(songs['ranks'].tolist())

    if method == 'corrcoef':
        covariance_matrix = np.corrcoef(songs_mat)
    elif method == 'cov':
        covariance_matrix = np.cov(songs_mat)
    else:
        return "Please choose 'cov' or 'corrcoeff' for method."

    mask = np.zeros_like(covariance_matrix)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(covariance_matrix, mask=mask, cmap='RdBu_r')

    prefix = 'Normalized' if method == 'corrcoef' else ''
    plt.title(prefix + ' Covariance Matrix for Song Rankings Through Time',
              weight='bold')
    plt.xticks(np.arange(0, 560, 25))
    plt.yticks(np.arange(0, 560, 25))
    plt.xlabel('Song Number', weight='bold')
    plt.ylabel('Song Number', weight='bold')
    if save:
        if out_as == 'pdf':
            plt.savefig(fig_path + method + '_matrix.pdf')
        elif out_as == 'png':
            plt.savefig(fig_path + method + '_matrix.png', dpi=300)
        else:
            print "Not saving; please choose 'png' or 'pdf' for 'out_as'"
    plt.show()
    return covariance_matrix

covars = get_covariances(method='corrcoef', save=True, out_as='pdf')




# We can query that covariance matrix now nicely.
# Let's see which songs' fates are tied least and most closely to Africa.

def find_sim_or_diff(cov_mat=covars, song='178'):
    """This doesn't work rn."""
    song = '178'
    s = get_page_idx(song)
    other_songs = range(557)
    other_songs.pop(s)
    covs = [covars[s][s2] for s2 in other_songs]
    np.where(covs == max(covs))[0][0]
    most = np.where(covs == max(covs))[0][0]
    least = np.where(covs == min(covs))[0][0]
    closest = [songs_df['page'][s], songs_df['page'][most]]
    farthest = [songs_df['page'][s], songs_df['page'][least]]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_songs_popularity(closest) #, fig=fig, ax=ax1)
    plot_songs_popularity(farthest) #, fig=fig, ax=ax2)
    # fig.show()




s1 = 0
covs_100 = [covars[s1][s2] for s2 in range(1, 557)]
np.where(covs_100 == max(covs_100))[0][0]


plot_songs_popularity([songs_df['page'][0], songs_df['page'][4]])













# The End
