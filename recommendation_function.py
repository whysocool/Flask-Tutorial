import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from matplotlib.pyplot import MultipleLocator

K = 9  # optimal k for kmeans


# data preprocess. After preprocessing, only 136 entries are available
# drop nan
df = pd.read_csv("./Travel details dataset.csv").dropna(axis=0, how='any')  # 136 valid rows
df = df.reset_index(drop=True)


# convert nation to nationality in nationality column
def clean_nationality(df, column_name):
    nation_list = ['Italy', 'China', 'Canada', 'South Korea', 'USA', 'Spain', 'Japan', 'Brazil', 'Germany',
                   'United Kingdom', 'Hong Kong', 'Singapore', 'Greece', 'Cambodia']
    nationality_list = ['Italian', 'Chinese', 'Canadian', 'South Korean', 'American', 'Spanish', 'Japanese',
                        'Brazilian',
                        'German',
                        'UK', 'Hong Kongese', 'Singaporean', 'Greek', 'Cambodian']
    column_nationality = df[column_name].copy()
    for i in range(len(column_nationality)):
        if column_nationality[i] in nation_list:
            column_nationality[i] = nationality_list[nation_list.index(column_nationality[i])]
    return column_nationality


df['Traveler nationality'] = clean_nationality(df, 'Traveler nationality')


# clear string or symbol in column 'Accommodation cost' and 'Transportation cost'
def extract_digit_for_column(df, column_name):
    column_new = df[column_name].copy()
    for i in range(len(column_new)):
        string_now = column_new[i]
        num = ""
        for c in string_now:
            if c.isdigit():
                num = num + c
        column_new[i] = num
    return column_new


df['Accommodation cost'] = extract_digit_for_column(df, 'Accommodation cost')
df['Transportation cost'] = extract_digit_for_column(df, 'Transportation cost')


# remove country from destination
def remove_country_from_destination(df, column_name):
    column_new = df[column_name].copy()
    for i in range(len(column_new)):
        column_new[i] = column_new[i].split(', ')[0]
    return column_new


df['Destination'] = remove_country_from_destination(df, 'Destination')


# change New York City to New York
def remove_duplicate_destination(df, column_name):
    # New York City -> New York
    column_Destination = df[column_name].copy()
    for i in range(len(column_Destination)):
        if column_Destination[i] == 'New York City':
            column_Destination[i] = 'New York'
    return column_Destination


df['Destination'] = remove_duplicate_destination(df, 'Destination')


# normalize age, accommodation cost, transportation cost
def get_scaler_and_column_number(df, column_name):
    column_number = df[column_name].values.reshape(-1, 1)
    scaler_number = StandardScaler()
    new_column = scaler_number.fit_transform(column_number)
    return scaler_number, new_column


# one hot gender, nationality
def get_scaler_and_column_str(df, column_name):
    column_str = df[column_name]
    scaler_str = LabelBinarizer()
    new_column = scaler_str.fit_transform(column_str)
    return scaler_str, new_column


# k=9,  because compared to accuracy, we want more data to be shown to users.
def find_optimal_k(x_array):
    inertia = []
    sil = []
    for k in range(2, 130):
        kmeans = KMeans(init='k-means++', n_clusters=k, random_state=0, n_init=1).fit(x_array)
        inertia.append(kmeans.inertia_)
        sil.append(silhouette_score(x_array, kmeans.labels_, metric='euclidean'))

    plt.plot(range(2, 130), inertia)
    plt.title('inertia')
    plt.show()

    x_major_locator = MultipleLocator(2)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.plot(range(2, 130), sil)
    plt.title('silhouette_score')
    plt.show()
    return


def return_kmeans_scaler_groupdct(df):
    scaler_age, column_age = get_scaler_and_column_number(df, 'Traveler age')
    scaler_accommodation, column_accommodation = get_scaler_and_column_number(df, 'Accommodation cost')
    scaler_transportation, column_transportation = get_scaler_and_column_number(df, 'Transportation cost')
    scaler_gender, column_gender = get_scaler_and_column_str(df, 'Traveler gender')
    scaler_nationality, column_nationality = get_scaler_and_column_str(df, 'Traveler nationality')

    nd_array_all_5attributes = np.concatenate(
        (column_age, column_gender, column_nationality, column_accommodation, column_transportation), axis=1)

    kmeans = KMeans(init='k-means++', n_clusters=9, random_state=0, n_init=1).fit(nd_array_all_5attributes)

    group_no = kmeans.labels_.reshape(-1, 1)
    destination_column = df['Destination'].to_numpy().reshape(-1, 1)

    array_destination_group = np.concatenate((destination_column, group_no), axis=1)
    group_dct = {}
    for i in range(K):
        group_dct[i] = []
    for i in range(len(array_destination_group)):
        current_group = array_destination_group[i, 1]
        current_destination = array_destination_group[i, 0]
        group_dct[current_group].append(current_destination)
    for i in range(K):
        destination_list = group_dct[i]
        dict_temp = {}
        for key in destination_list:
            dict_temp[key] = dict_temp.get(key, 0) + 1
        sorted_destinations = sorted(dict_temp.items(), key=lambda x: x[1], reverse=True)
        group_dct[i] = sorted_destinations
    return kmeans, scaler_age, scaler_gender, scaler_nationality, scaler_accommodation, scaler_transportation, group_dct


kmeans, scaler_age, scaler_gender, scaler_nationality, scaler_accommodation, scaler_transportation, group_dct = return_kmeans_scaler_groupdct(
    df)

################################################################### test
# only need age, gender, and nationality, accommodation cost, transportation cost
new_customer1 = [[35, 'Female', 'Korean', 1000, 600]]
df_new_customer1 = pd.DataFrame(new_customer1, columns=['Traveler age', 'Traveler gender', 'Traveler nationality',
                                                        'Accommodation cost', 'Transportation cost'])

column_age = df_new_customer1['Traveler age'].values.reshape(-1, 1)
column_age = scaler_age.transform(column_age)

column_gender = df_new_customer1['Traveler gender']
column_gender = scaler_gender.transform(column_gender)

column_nationality = df_new_customer1['Traveler nationality']
column_nationality = scaler_nationality.transform(column_nationality)

column_accommodation = df_new_customer1['Accommodation cost'].values.reshape(-1, 1)
column_accommodation = scaler_accommodation.transform(column_accommodation)

column_transportation = df_new_customer1['Transportation cost'].values.reshape(-1, 1)
column_transportation = scaler_transportation.transform(column_transportation)

ndarray_new_customer1 = np.concatenate(
    (column_age, column_gender, column_nationality, column_accommodation, column_transportation), axis=1)
group_of_new_customer1 = kmeans.predict(ndarray_new_customer1).item()
print(group_dct[group_of_new_customer1])
