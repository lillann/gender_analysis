
import json
import pandas as pd

from util import univs, langs, department_topics, topic_to_int

info_files = {}
for uni in univs:
    info_files[uni] = f"data/{uni}-metadata-210310.json.gz" 

# Read metadata files into dataframes
dfs = {}
for uni in univs:
    dfs[uni] = pd.read_json(info_files[uni]) 

# Create column for gender and filter out non-male/female 
genders = {}
for uni in univs:
    genders[uni] = [val['gender'] for val in dfs[uni]['inferred'].values]

for uni in univs:
    dfs[uni]['gender_composition'] = genders[uni]

for uni in univs:
    dfs[uni] = dfs[uni][dfs[uni]['gender_composition'].isin(['male', 'female'])]

# Separate data by language
languages = {}
for uni in univs:
    languages[uni] = [val['language'] for val in dfs[uni]['inferred'].values]

for uni in univs:
    dfs[uni]['language'] = languages[uni]

dfs2 = {}
for uni in univs:
    dfs2[uni] = {}
    for lang in langs:
        dfs2[uni][lang] = dfs[uni][dfs[uni]['language'] == lang]


# Get the number of female/male theses in the given department and dataframe
def get_sizes(df, dep):
    df_dep = df[df['department'] == dep]
    size_female, size_male = [len(df_dep[df_dep['gender_composition'] == gender]) 
                              for gender in ['female', 'male']] 
    return (size_female, size_male)

# Randomly downsamples data by reducing the majority class
# resulting in an equal number of male and female authors
# for each department.

def decide_fraction(df, dep):
    (a, b) = get_sizes(df, dep)
    if a == 0 or b == 0: 
        return ('', 1)
    elif a > b:
        return ('female', 1  - (b / a))
    else:
        return ('male', 1 - (a / b))
    

def drop_fraction(df, dep, gender, fraction):
    df2 = df[df.department.eq(dep)]
    df2 = df2[df2.gender_composition.eq(gender)]
    return df.drop(df2.sample(frac=fraction).index)

def gender_equal(df):
    deps = df['department'].unique()
    return gender_equal_deps(df, deps)


def gender_equal_deps(df, deps):
    for dep in deps:
        (gen, frac) = decide_fraction(df, dep)
        # print(dep, gen, frac)
        df = drop_fraction(df, dep, gen, frac)
    return df

df_equal = {}
for uni in univs:
    df_equal[uni] = {}
    for lang in langs:
        df_equal[uni][lang] = gender_equal(dfs2[uni][lang])
        print(f"Gender counts {uni} {lang}")
        print(df_equal[uni][lang]['gender_composition'].value_counts())

# Restrict to departments

df_final = {}
for lang in langs:
    df_final[lang] = pd.concat([
        df_equal[uni][lang][df_equal[uni][lang]['department'].isin(department_topics)]
        for uni in univs
    ])

for lang in langs:
    labels_topic = [department_topics[dep] for dep in df_final[lang]['department']]
    df_final[lang]['topic'] = labels_topic


for lang in langs:
    df_final[lang].to_csv(f"data/theses-{lang}-downsampled.csv.gz")
    # df_final[lang].to_json(f"data/theses-{lang}-downsampled.jsonl.gz", lines=True, orient="records")

