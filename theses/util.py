

univs = ['GU', 'CTH']

langs = ['sv', 'en']


# genders
genders = "female male".split()

gender_to_int_dict = {gen: n for n, gen in enumerate(genders, 1)}
int_to_gender_dict = {n: gen for n, gen in enumerate(genders, 1)}

def gender_to_int(gen):
    return gender_to_int_dict.get(gen)

def int_to_gender(n):
    return int_to_gender_dict.get(n)


# topics and departments
topic_departments = {}
topic_departments['humaniora'] = [
    'Institutionen för filosofi, lingvistik och vetenskapsteori',
    'Institutionen för litteratur, idéhistoria och religion',
    'Institutionen för språk och litteraturer',
    'Institutionen för svenska språket',
]
topic_departments['arkitektur'] = [
    'Institutionen för arkitektur och samhällsbyggnadsteknik',
]
topic_departments['ekonomi'] = [
    'Företagsekonomiska institutionen',
]
topic_departments['naturvetenskap'] = [
    'Institutionen för biologi och bioteknik',
    'Institutionen för fysik',
    'Institutionen för kemi och kemiteknik',
    'Institutionen för rymd- och geovetenskap',
]

department_topics = {dep : top for top in topic_departments for dep in topic_departments[top]}

topics = list(topic_departments)
topic_to_int_dict = {top: n for n, top in enumerate(topics, 1)}
int_to_topic_dict = {n: top for n, top in enumerate(topics, 1)}

def topic_to_int(top):
    return topic_to_int_dict.get(top)

def int_to_topic(n):
    return int_to_topic_dict.get(n)


from sklearn.preprocessing import StandardScaler
import numpy as np

def scale_frequencies(freqs):
    scaler = StandardScaler()   
    scaler.fit(freqs)
    freqs_std = list(scaler.fit_transform(freqs))
    freq_std = np.nan_to_num(freqs_std) 
    return freq_std
