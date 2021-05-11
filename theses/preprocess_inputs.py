from sklearn.preprocessing import StandardScaler
import numpy as np

# Divide the departments into four main topics
humaniora = ['Institutionen för filosofi, lingvistik och vetenskapsteori','Institutionen för litteratur, idéhistoria och religion','Institutionen för språk och litteraturer','Institutionen för svenska språket']
arkitektur  = ['Institutionen för arkitektur och samhällsbyggnadsteknik']
ekonomi = ['Företagsekonomiska institutionen']
naturvetenskap = ['Institutionen för biologi och bioteknik','Institutionen för fysik','Institutionen för kemi och kemiteknik','Institutionen för rymd- och geovetenskap']

def scale_frequencies(freqs) : 
    scaler = StandardScaler()   
    scaler.fit(freqs)
    freqs_std = list(scaler.fit_transform(freqs))
    freq_std = np.nan_to_num(freqs_std) 
    return freq_std
    
    
# Code gender and topic with integer value
def gender_to_int(gender) :
    if gender == 'female' : return 0
    else : return 1 

 
def topic_to_int(dept) :
 
    if dept in humaniora : return 0 
    elif dept in arkitektur : return 1  
    elif dept in ekonomi : return 2 
    else : return 3 