import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
import pickle


DATA_URL = (
    "advert-data-to-combine.csv"
)

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

def consolidate(new_idea, data):
    consolidated = data.append(new_idea)
    return consolidated

def matrix_description(consolidated):
    cv = CountVectorizer()
    cv.fit(consolidated['Idea_Description'])
    results = cv.transform(consolidated['Idea_Description'])
    features = cv.get_feature_names()
    df_res = pd.DataFrame(results.toarray(), columns=features)
    df_res.columns = [str(col) + '_in_description' for col in df_res.columns]
    return df_res

def matrix_header(consolidated):
    cv = CountVectorizer()
    cv.fit(consolidated['Idea_Name'])
    results = cv.transform(consolidated['Idea_Name'])
    features = cv.get_feature_names()
    df_res2 = pd.DataFrame(results.toarray(), columns=features)
    df_res2.columns = [str(col) + '_in_header' for col in df_res2.columns]
    return df_res2

def one_hot_encode(consolidated, df_res, df_res2):
    onehot_df = pd.get_dummies(consolidated.Capitals)
    onehot_df5 = pd.get_dummies(consolidated.Layout)
    onehot_df6 = pd.get_dummies(consolidated.Headline_Positioning)
    onehot_df9 = pd.get_dummies(consolidated.Body_Font)
    onehot_df10 = pd.get_dummies(consolidated.Heading_Font)
    onehot_df11 = pd.get_dummies(consolidated.Image_Theme)
    final_df = pd.concat([consolidated, onehot_df, onehot_df5, onehot_df6, onehot_df9, onehot_df10, onehot_df11], axis=1)
    return final_df

def all_together(final_df, df_res, df_res2):
    final_df.reset_index(drop=True, inplace=True)
    final_df = pd.concat([final_df, df_res, df_res2], axis=1)
    final_df.drop_duplicates(inplace=True)
    return final_df

def predict_uniqueness(final_df):
    filename = 'finalized_unique_model-diff-features.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    classifications = loaded_model.predict(final_df[['better_in_description', 'planet_in_description', 'stimulates_in_description', 'better_in_header', 'deep_in_header', 'moisture_in_header', 'oil_in_header', 'almond_in_header', 'lemon_in_header', 'meringue_in_header', 'key_in_header', 'lime_in_header', 'macadamia_in_header', 'salted_in_header', 'mask_in_header', 'pomegranate_in_header', 'Layout_3', 'planet_in_header', 'avocado_in_header', 'calendula_in_header', 'extract_in_header', 'raspberry_in_header', 'vanilla_in_header', 'spice_in_header', 'cream_in_header', 'creamy_in_description', 'different_in_description', 'drop_in_description', 'none_in_description', 'rich_in_description', 'transforms_in_description', 'warm_in_description', 'way_in_description', 'shower_in_header', 'dissolve_in_description', 'nourishs_in_description', 'water_in_description', 'cleanse_in_description', 'contain_in_description', 'friendly_in_description', 'pocket_in_description', 'travelling_in_description', 'burnt_in_header', 'hand_in_header', 'sage_in_header', 'cacao_in_header', 'salt_in_header', 'honey_in_header', 'caramel_in_header', 'lavender_in_header', 'based_in_description', 'white_in_header', 'bottle_in_description', 'sea_in_header', 'creamsicle_in_header', 'mint_in_header', 'detox_in_header', 'use_in_description', 'prevent_in_description', 'prone_in_description', 'root_in_description', 'soothing_in_description', 'special_in_description', 'witchhazel_in_description', 'botanica_in_header', 'made_in_description', 'glycerine_in_description', 'revitalise_in_description', 'glycerine_in_header', 'body_in_header', 'toasted_in_header', 'power_in_description', 'look_in_description', 'roasted_in_header', 'certified_in_description', 'unleash_in_description', 'Top_Left_Heading_Position', 'feel_in_header', 'eggnog_in_header', 'whipped_in_header', 'Recycle_Image_Theme', 'clean_in_description', 'butterscotch_in_header', 'pear_in_header', 'day_in_description', 'good_in_description', 'great_in_description', 'reviving_in_description', 'brand_in_header', 'Layout_8', 'Bottom_Right_Heading_Position', 'Happiness_Image_Theme', 'next_in_description', 'ready_in_description', 'whatever_in_description', 'double_in_description', 'dry_in_description', 'goodbye_in_description', 'moisture_in_description', 'moisturizing_in_description', 'regular_in_description', 'rough_in_description', 'say_in_description', 'wash_in_description', 'butter_in_header', 'double_in_header', 'cleansing_in_header', 'custard_in_header', 'cleaner_in_description', 'dirtier_in_description', 'feel_in_description',  'cleaner_in_header', 'dirtier_in_header', 'Bottles_Image_Theme', 'deodorize_in_description', 'moist_in_description', 'refresh_in_description', 'super_in_description', 'care_in_header']])
    final_df["Uniqueness_Prediction"]  = classifications
    return final_df

def predict_appeal(final_df):
    filename = 'finalized_appeal_model-diff-features.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    classifications = loaded_model.predict(final_df[['better_in_description', 'planet_in_description', 'stimulates_in_description', 'better_in_header', 'deep_in_header', 'moisture_in_header', 'oil_in_header', 'almond_in_header', 'lemon_in_header', 'meringue_in_header', 'key_in_header', 'lime_in_header', 'macadamia_in_header', 'salted_in_header', 'mask_in_header', 'pomegranate_in_header', 'Layout_3', 'planet_in_header', 'avocado_in_header', 'calendula_in_header', 'extract_in_header', 'raspberry_in_header', 'vanilla_in_header', 'spice_in_header', 'cream_in_header', 'creamy_in_description', 'different_in_description', 'drop_in_description', 'none_in_description', 'rich_in_description', 'transforms_in_description', 'warm_in_description', 'way_in_description', 'shower_in_header', 'dissolve_in_description', 'nourishs_in_description', 'water_in_description', 'cleanse_in_description', 'contain_in_description', 'friendly_in_description', 'pocket_in_description', 'travelling_in_description', 'burnt_in_header', 'hand_in_header', 'sage_in_header', 'cacao_in_header', 'salt_in_header', 'honey_in_header', 'caramel_in_header', 'lavender_in_header', 'based_in_description', 'white_in_header', 'bottle_in_description', 'sea_in_header', 'creamsicle_in_header', 'mint_in_header', 'detox_in_header', 'use_in_description', 'prevent_in_description', 'prone_in_description', 'root_in_description', 'soothing_in_description', 'special_in_description', 'witchhazel_in_description', 'botanica_in_header', 'made_in_description', 'glycerine_in_description', 'revitalise_in_description', 'glycerine_in_header', 'body_in_header', 'toasted_in_header', 'power_in_description', 'look_in_description', 'roasted_in_header', 'certified_in_description', 'unleash_in_description', 'Top_Left_Heading_Position', 'feel_in_header', 'eggnog_in_header', 'whipped_in_header', 'Recycle_Image_Theme', 'clean_in_description', 'butterscotch_in_header', 'pear_in_header', 'day_in_description', 'good_in_description', 'great_in_description', 'reviving_in_description', 'brand_in_header', 'Layout_8', 'Bottom_Right_Heading_Position', 'Happiness_Image_Theme', 'next_in_description', 'ready_in_description', 'whatever_in_description', 'double_in_description', 'dry_in_description', 'goodbye_in_description', 'moisture_in_description', 'moisturizing_in_description', 'regular_in_description', 'rough_in_description', 'say_in_description', 'wash_in_description', 'butter_in_header', 'double_in_header', 'cleansing_in_header', 'custard_in_header', 'cleaner_in_description', 'dirtier_in_description', 'feel_in_description', 'cleaner_in_header', 'dirtier_in_header', 'Bottles_Image_Theme', 'deodorize_in_description', 'moist_in_description', 'refresh_in_description', 'super_in_description', 'care_in_header']])
    final_df["Bet_For_Prediction"]  = classifications
    return final_df

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
image = Image.open('Unilever.png')
st.sidebar.image(image)
#st.sidebar.markdown("<h1 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2.1rem'><strong style='text-align:center;display:block;color:#0D009D'>U-LEARN</strong>", unsafe_allow_html=True)
st.text("")
st.sidebar.title("Choose an option")
st.text("")
st.sidebar.markdown("### What do you want to do?")
select = st.sidebar.selectbox('Choose activity', ['Explore stimulus learnings', 'Rate my stimulus'], key='1')
st.text("")


if select =='Explore stimulus learnings':
    st.sidebar.markdown("### What do you want to learn about?")
    radio = st.sidebar.radio("Choose stimulus element", ('Use of language in heading', 'Use of language in description', 'Stimulus design elements'), key='2')
    if radio == 'Use of language in description':
        st.title("Use of language in the stimulus description")
        st.text("")
        st.markdown("<p style='font-weight:normal'>This section explores the impact of words included within the <strong>stimulus description</strong> on both the <strong>UNIQUENESS</strong> and <strong>BET FOR</strong> scores for a stimulus.</p>", unsafe_allow_html=True)
        st.text("")
        st.markdown("<h2 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Positive description words</strong></h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:normal'>The greater the score, the <strong>more likely</strong> the term is to appear in <strong>the description</strong> of stimuli that have both a relatively <strong>HIGH UNIQUENESS</strong> and <strong>HIGH BET FOR</strong> score.</p>", unsafe_allow_html=True)
        st.text("")
        positive_description_categories = [
    {"name": "Positive impact of the word 'stimulates' in description", "value":37},
    {"name": "Positive impact of the word 'pocket' in description", "value":36},
    {"name": "Positive impact of the word 'nourishes' in description", "value":35},
    {"name": "Positive impact of the word 'dissolve' in description", "value":34},
    {"name": "Positive impact of the word 'pack' in description", "value":34},
    {"name": "Positive impact of the word 'strip' in description", "value":34},
    {"name": "Positive impact of the word 'travelling' in description", "value":34},
    {"name": "Positive impact of the word 'filler' in description", "value":32},
    {"name": "Positive impact of the word 'creamy' in description", "value":32},
    {"name": "Positive impact of the word 'transforms' in description", "value":32},
    {"name": "Positive impact of the word 'friendly' in description", "value":31},
    {"name": "Positive impact of the word 'leaf' in description", "value":30},
    {"name": "Positive impact of the word 'contain' in description", "value":29},
    {"name": "Positive impact of the word 'warm' in description", "value":28},
    {"name": "Positive impact of the word 'drop' in description", "value":26},
    {"name": "Positive impact of the word 'none' in description", "value":24},
    {"name": "Positive impact of the word 'sense' in description", "value":22},
    {"name": "Positive impact of the word 'rich' in description", "value":19},
    {"name": "Positive impact of the word 'planet' in description", "value":18},
    {"name": "Positive impact of the word 'exfoliating' in description", "value":14},
    {"name": "Positive impact of the word 'customized' in description", "value":14},
    {"name": "Positive impact of the word 'heel' in description", "value":14},
    {"name": "Positive impact of the word 'scrub' in description", "value":14},
    {"name": "Positive impact of the word 'hand' in description", "value":14},



        ]

        positive_description_subplots = make_subplots(
            rows=len(positive_description_categories),
            cols=1,
            subplot_titles=[x["name"] for x in positive_description_categories],
            shared_xaxes=True,
            print_grid=False,
            vertical_spacing=(0.45 / len(positive_description_categories)),
        )
        _ = positive_description_subplots['layout'].update(
            width=550,
            plot_bgcolor='#fff',
        )

        for k, x in enumerate(positive_description_categories):
            positive_description_subplots.add_trace(dict(
                type='bar',
                orientation='h',
                y=[x["name"]],
                x=[x["value"]],
                text=["{:,.0f}".format(x["value"])],
                hoverinfo='text',
                textposition='auto',
                marker=dict(
                    color="#047508",
                ),
            ), k+1, 1)

            positive_description_subplots['layout'].update(
            showlegend=False,
        )
        for x in positive_description_subplots["layout"]['annotations']:
            x['x'] = 0
            x['xanchor'] = 'left'
            x['align'] = 'left'
            x['font'] = dict(
                size=12,
            )

        for axis in positive_description_subplots['layout']:
            if axis.startswith('yaxis') or axis.startswith('xaxis'):
                positive_description_subplots['layout'][axis]['visible'] = False

        positive_description_subplots['layout']['margin'] = {
            'l': 0,
            'r': 0,
            't': 20,
            'b': 1,
        }
        height_calc = 45 * len(positive_description_categories)
        height_calc = max([height_calc, 350])
        positive_description_subplots['layout']['height'] = height_calc
        positive_description_subplots['layout']['width'] = 700

        positive_description_subplots.update_xaxes(range=[0, 80])


        st.plotly_chart(positive_description_subplots)

        st.markdown("<h2 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Negative description words</strong></h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:normal'>The lower the score, the <strong>more likely</strong> the term is to appear in <strong>the description</strong> of stimuli that have both a <strong>LOW UNIQUENESS</strong> and <strong>LOW BET FOR</strong> score.</p>", unsafe_allow_html=True)
        st.text("")
        negative_description_categories = [
    {"name": "Negative impact of the word 'deodorize' in description", "value":84},
    {"name": "Negative impact of the word 'wipe' in description", "value":83},
    {"name": "Negative impact of the word 'moist' in description", "value":80},
    {"name": "Negative impact of the word 'wash' in description", "value":76},
    {"name": "Negative impact of the word 'moisturizing' in description", "value":76},
    {"name": "Negative impact of the word 'goodbye' in description", "value":75},
    {"name": "Negative impact of the word 'reviving' in description", "value":74},
    {"name": "Negative impact of the word 'butter' in description", "value":74},
    {"name": "Negative impact of the word 'rough' in description", "value":71},
    {"name": "Negative impact of the word 'whatever' in description", "value":71},
    {"name": "Negative impact of the word 'refresh' in description", "value":69},
    {"name": "Negative impact of the word 'dry' in description", "value":68},
    {"name": "Negative impact of the word 'unleash' in description", "value":67},
    {"name": "Negative impact of the word 'ready' in description", "value":67},
    {"name": "Negative impact of the word 'clean' in description", "value":66},
    {"name": "Negative impact of the word 'glycerine' in description", "value":64},
    {"name": "Negative impact of the word 'revitalise' in description", "value":64},
    {"name": "Negative impact of the word 'condition' in description", "value":64},
    {"name": "Negative impact of the word 'herbal' in description", "value":62},
    {"name": "Negative impact of the word 'certified' in description", "value":60},
    {"name": "Negative impact of the word 'mineral' in description", "value":60},
    {"name": "Negative impact of the word 'vegan' in description", "value":60},
    {"name": "Negative impact of the word 'radox' in description", "value":59},
    {"name": "Negative impact of the word 'rejuvenated' in description", "value":59},







        ]

        negative_description_subplots = make_subplots(
            rows=len(negative_description_categories),
            cols=1,
            subplot_titles=[x["name"] for x in negative_description_categories],
            shared_xaxes=True,
            print_grid=False,
            vertical_spacing=(0.45 / len(negative_description_categories)),
        )
        _ = negative_description_subplots['layout'].update(
            width=550,
            plot_bgcolor='#fff',
        )

        for k, x in enumerate(negative_description_categories):
            negative_description_subplots.add_trace(dict(
                type='bar',
                orientation='h',
                y=[x["name"]],
                x=[x["value"]],
                text=["{:,.0f}".format(x["value"])],
                hoverinfo='text',
                textposition='auto',
                marker=dict(
                    color="#8B0000",
                ),
            ), k+1, 1)

            negative_description_subplots['layout'].update(
            showlegend=False,
        )
        for x in negative_description_subplots["layout"]['annotations']:
            x['x'] = 0
            x['xanchor'] = 'left'
            x['align'] = 'left'
            x['font'] = dict(
                size=12,
            )

        for axis in negative_description_subplots['layout']:
            if axis.startswith('yaxis') or axis.startswith('xaxis'):
                negative_description_subplots['layout'][axis]['visible'] = False

        negative_description_subplots['layout']['margin'] = {
            'l': 0,
            'r': 0,
            't': 20,
            'b': 1,
        }
        height_calc = 45 * len(negative_description_categories)
        height_calc = max([height_calc, 350])
        negative_description_subplots['layout']['height'] = height_calc
        negative_description_subplots['layout']['width'] = 700

        negative_description_subplots.update_xaxes(range=[10, 90])


        st.plotly_chart(negative_description_subplots)

    if radio == 'Use of language in heading':
        st.title("Use of language in the stimulus heading")
        st.text("")
        st.markdown("<p style='font-weight:normal'>This section explores the impact of words included within the <strong>stimulus heading</strong> on both the <strong>UNIQUENESS</strong> and <strong>BET FOR</strong> scores for a stimulus.</p>", unsafe_allow_html=True)
        st.text("")
        st.markdown("<h2 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Positive heading words</strong></h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:normal'>The greater the score, the <strong>more likely</strong> the term is to appear in <strong>the heading</strong> of stimuli that have both a relatively <strong>HIGH UNIQUENESS</strong> and <strong>HIGH BET FOR</strong> score.</p>", unsafe_allow_html=True)
        st.text("")
        positive_header_categories = [
    {"name": "Positive impact of the word 'almond' in header", "value":45},
    {"name": "Positive impact of the word 'macadamia' in header", "value":43},
    {"name": "Positive impact of the word 'raspberry' in header", "value":42},
    {"name": "Positive impact of the word 'vanilla' in header", "value":41},
    {"name": "Positive impact of the word 'lime' in header", "value":41},
    {"name": "Positive impact of the word 'salted' in header", "value":41},
    {"name": "Positive impact of the word 'pomegranate' in header", "value":40},
    {"name": "Positive impact of the word 'mask' in header", "value":37},
    {"name": "Positive impact of the word 'enzyme' in header", "value":36},
    {"name": "Positive impact of the word 'burnt' in header", "value":35},
    {"name": "Positive impact of the word 'sage' in header", "value":35},
    {"name": "Positive impact of the word 'strip' in header", "value":34},
    {"name": "Positive impact of the word 'oil' in header", "value":32},
    {"name": "Positive impact of the word 'shower' in header", "value":30},
    {"name": "Positive impact of the word 'brulee' in header", "value":28},
    {"name": "Positive impact of the word 'creme' in header", "value":28},
    {"name": "Positive impact of the word 'matcha' in header", "value":28},
    {"name": "Positive impact of the word 'sakura' in header", "value":28},
    {"name": "Positive impact of the word 'mint' in header", "value":26},
    {"name": "Positive impact of the word 'drop' in header", "value":26},
    {"name": "Positive impact of the word 'moisture' in header", "value":25},
    {"name": "Positive impact of the word 'planet' in header", "value":19},
    {"name": "Positive impact of the word 'deep' in header", "value":17},
    {"name": "Positive impact of the word 'key' in header", "value":14},
    {"name": "Positive impact of the word 'hand' in header", "value":14},





        ]

        positive_header_subplots = make_subplots(
            rows=len(positive_header_categories),
            cols=1,
            subplot_titles=[x["name"] for x in positive_header_categories],
            shared_xaxes=True,
            print_grid=False,
            vertical_spacing=(0.45 / len(positive_header_categories)),
        )
        _ = positive_header_subplots['layout'].update(
            width=550,
            plot_bgcolor='#fff',
        )

        for k, x in enumerate(positive_header_categories):
            positive_header_subplots.add_trace(dict(
                type='bar',
                orientation='h',
                y=[x["name"]],
                x=[x["value"]],
                text=["{:,.0f}".format(x["value"])],
                hoverinfo='text',
                textposition='auto',
                marker=dict(
                    color="#047508",
                ),
            ), k+1, 1)

            positive_header_subplots['layout'].update(
            showlegend=False,
        )
        for x in positive_header_subplots["layout"]['annotations']:
            x['x'] = 0
            x['xanchor'] = 'left'
            x['align'] = 'left'
            x['font'] = dict(
                size=12,
            )

        for axis in positive_header_subplots['layout']:
            if axis.startswith('yaxis') or axis.startswith('xaxis'):
                positive_header_subplots['layout'][axis]['visible'] = False

        positive_header_subplots['layout']['margin'] = {
            'l': 0,
            'r': 0,
            't': 20,
            'b': 1,
        }
        height_calc = 45 * len(positive_header_categories)
        height_calc = max([height_calc, 350])
        positive_header_subplots['layout']['height'] = height_calc
        positive_header_subplots['layout']['width'] = 700

        positive_header_subplots.update_xaxes(range=[0, 80])

        st.plotly_chart(positive_header_subplots)

        st.markdown("<h2 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Negative heading words</strong></h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:normal'>The lower the score, the <strong>more likely</strong> the term is to appear in <strong>the heading</strong> of stimuli that have both a <strong>LOW UNIQUENESS</strong> and <strong>LOW BET FOR</strong> score.</p>", unsafe_allow_html=True)
        st.text("")
        negative_header_categories = [
    {"name": "Negative impact of the word 'wipe' in header", "value":83},
    {"name": "Negative impact of the word 'cleansing' in header", "value":78},
    {"name": "Negative impact of the word 'butter' in header", "value":74},
    {"name": "Negative impact of the word 'glycerine' in header", "value":64},
    {"name": "Negative impact of the word 'toasted' in header", "value":64},
    {"name": "Negative impact of the word 'cappuccino' in header", "value":62},
    {"name": "Negative impact of the word 'mocha' in header", "value":62},
    {"name": "Negative impact of the word 'herb' in header", "value":62},
    {"name": "Negative impact of the word 'mineral' in header", "value":60},
    {"name": "Negative impact of the word 'blend' in header", "value":59},
    {"name": "Negative impact of the word 'recycle' in header", "value":59},
    {"name": "Negative impact of the word 'pistachio' in header", "value":59},
    {"name": "Negative impact of the word 'feel' in header", "value":57},
    {"name": "Negative impact of the word 'pumpkin' in header", "value":57},
    {"name": "Negative impact of the word 'herbal' in header", "value":56},
    {"name": "Negative impact of the word 'enjoy' in header", "value":55},
    {"name": "Negative impact of the word 'spiced' in header", "value":55},
    {"name": "Negative impact of the word 'double' in header", "value":52},
    {"name": "Negative impact of the word 'care' in header", "value":52},
    {"name": "Negative impact of the word 'wash' in header", "value":49},
    {"name": "Negative impact of the word 'share' in header", "value":49},
    {"name": "Negative impact of the word 'organic' in header", "value":49},
    {"name": "Negative impact of the word 'fantastic' in header", "value":48},
    {"name": "Negative impact of the word 'unique' in header", "value":47},
    {"name": "Negative impact of the word 'happy' in header", "value":47},


        ]

        negative_header_subplots = make_subplots(
            rows=len(negative_header_categories),
            cols=1,
            subplot_titles=[x["name"] for x in negative_header_categories],
            shared_xaxes=True,
            print_grid=False,
            vertical_spacing=(0.45 / len(negative_header_categories)),
        )
        _ = negative_header_subplots['layout'].update(
            width=550,
            plot_bgcolor='#fff',
        )

        for k, x in enumerate(negative_header_categories):
            negative_header_subplots.add_trace(dict(
                type='bar',
                orientation='h',
                y=[x["name"]],
                x=[x["value"]],
                text=["{:,.0f}".format(x["value"])],
                hoverinfo='text',
                textposition='auto',
                marker=dict(
                    color="#8B0000",
                ),
            ), k+1, 1)

            negative_header_subplots['layout'].update(
            showlegend=False,
        )
        for x in negative_header_subplots["layout"]['annotations']:
            x['x'] = 0
            x['xanchor'] = 'left'
            x['align'] = 'left'
            x['font'] = dict(
                size=12,
            )

        for axis in negative_header_subplots['layout']:
            if axis.startswith('yaxis') or axis.startswith('xaxis'):
                negative_header_subplots['layout'][axis]['visible'] = False

        negative_header_subplots['layout']['margin'] = {
            'l': 0,
            'r': 0,
            't': 20,
            'b': 1,
        }
        height_calc = 45 * len(negative_header_categories)
        height_calc = max([height_calc, 350])
        negative_header_subplots['layout']['height'] = height_calc
        negative_header_subplots['layout']['width'] = 700

        negative_header_subplots.update_xaxes(range=[0, 90])


        st.plotly_chart(negative_header_subplots)

    if radio == 'Stimulus design elements':
        st.title("Stimulus design elements")
        st.text("")
        st.markdown("<p style='font-weight:normal'>This section explores the impact of words included within the <strong>stimulus design</strong> on both the <strong>UNIQUENESS</strong> and <strong>BET FOR</strong> scores for a stimulus.</p>", unsafe_allow_html=True)
        st.text("")
        st.markdown("<h2 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Positive design elements</strong></h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:normal'>The greater the score, the <strong>more likely</strong> the <strong>design element</strong> is to feature in stimuli that have both a relatively <strong>HIGH UNIQUENESS</strong> and <strong>HIGH BET FOR</strong> score.</p>", unsafe_allow_html=True)
        st.text("")
        positive_design_categories = [
    {"name": "Heading contains more than 70 characters", "value":60},
    {"name": "Image theme is 'product'", "value":43},
    {"name": "Image theme is 'food'", "value":41},
    {"name": "Heading is in top central position", "value":39},
    {"name": "Description is Century Gothic font", "value":39},
    {"name": "Heading is Century Gothic font", "value":39},
    {"name": "Description is Century Gothic font", "value":39},
    {"name": "Heading contains 50 to 60 characters", "value":33},
    {"name": "Heading contains 60 to 70 characters", "value":33},
    {"name": "The stimulus contains 3 images", "value":31},


        ]

        positive_design_subplots = make_subplots(
            rows=len(positive_design_categories),
            cols=1,
            subplot_titles=[x["name"] for x in positive_design_categories],
            shared_xaxes=True,
            print_grid=False,
            vertical_spacing=(0.45 / len(positive_design_categories)),
        )
        _ = positive_design_subplots['layout'].update(
            width=550,
            plot_bgcolor='#fff',
        )

        for k, x in enumerate(positive_design_categories):
            positive_design_subplots.add_trace(dict(
                type='bar',
                orientation='h',
                y=[x["name"]],
                x=[x["value"]],
                text=["{:,.0f}".format(x["value"])],
                hoverinfo='text',
                textposition='auto',
                marker=dict(
                    color="#047508",
                ),
            ), k+1, 1)

            positive_design_subplots['layout'].update(
            showlegend=False,
        )
        for x in positive_design_subplots["layout"]['annotations']:
            x['x'] = 0
            x['xanchor'] = 'left'
            x['align'] = 'left'
            x['font'] = dict(
                size=12,
            )

        for axis in positive_design_subplots['layout']:
            if axis.startswith('yaxis') or axis.startswith('xaxis'):
                positive_design_subplots['layout'][axis]['visible'] = False

        positive_design_subplots['layout']['margin'] = {
            'l': 0,
            'r': 0,
            't': 20,
            'b': 1,
        }
        height_calc = 45 * len(positive_design_categories)
        height_calc = max([height_calc, 350])
        positive_design_subplots['layout']['height'] = height_calc
        positive_design_subplots['layout']['width'] = 700

        positive_design_subplots.update_xaxes(range=[0, 70])

        st.plotly_chart(positive_design_subplots)

        st.markdown("<h2 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Negative design elements</strong></h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:normal'>The lower the score, the <strong>more likely</strong> the <strong>design element</strong> is to feature in stimuli that have both a <strong>LOW UNIQUENESS</strong> and <strong>LOW BET FOR</strong> score.</p>", unsafe_allow_html=True)
        st.text("")
        negative_design_categories = [
    {"name": "Image theme is 'sun'", "value":100},
    {"name": "Image theme is 'recycling'", "value":100},
    {"name": "Image theme is 'happiness'", "value":100},
    {"name": "Heading font is Bahnschrift", "value":100},
    {"name": "Description font is Calibri Light", "value":79},
    {"name": "Heading contains fewer than 40 characters", "value":74},
    {"name": "Heading position is top right", "value":70},
    {"name": "Description font is Calibri", "value":67},
    {"name": "Heading case is Capitals", "value":60},
    {"name": "Image theme is 'plants'", "value":50},





        ]

        negative_design_subplots = make_subplots(
            rows=len(negative_design_categories),
            cols=1,
            subplot_titles=[x["name"] for x in negative_design_categories],
            shared_xaxes=True,
            print_grid=False,
            vertical_spacing=(0.45 / len(negative_design_categories)),
        )
        _ = negative_design_subplots['layout'].update(
            width=550,
            plot_bgcolor='#fff',
        )

        for k, x in enumerate(negative_design_categories):
            negative_design_subplots.add_trace(dict(
                type='bar',
                orientation='h',
                y=[x["name"]],
                x=[x["value"]],
                text=["{:,.0f}".format(x["value"])],
                hoverinfo='text',
                textposition='auto',
                marker=dict(
                    color="#8B0000",
                ),
            ), k+1, 1)

            negative_design_subplots['layout'].update(
            showlegend=False,
        )
        for x in negative_design_subplots["layout"]['annotations']:
            x['x'] = 0
            x['xanchor'] = 'left'
            x['align'] = 'left'
            x['font'] = dict(
                size=12,
            )

        for axis in negative_design_subplots['layout']:
            if axis.startswith('yaxis') or axis.startswith('xaxis'):
                negative_design_subplots['layout'][axis]['visible'] = False

        negative_design_subplots['layout']['margin'] = {
            'l': 0,
            'r': 0,
            't': 20,
            'b': 1,
        }
        height_calc = 45 * len(negative_design_categories)
        height_calc = max([height_calc, 350])
        negative_design_subplots['layout']['height'] = height_calc
        negative_design_subplots['layout']['width'] = 700

        negative_design_subplots.update_xaxes(range=[0, 100])

        st.plotly_chart(negative_design_subplots)

if select =='Rate my stimulus':
    st.sidebar.markdown("### How do you want to rate your stimulus?")
    st.title("Rate my stimulus")
    st.text("")
    st.markdown("<p style='font-weight:normal'>Choose an option from the <strong>menu on the left</strong> to upload your stimulus, to get a predicted <strong>UNIQUENESS</strong> and <strong>BET FOR</strong> rating.</p>", unsafe_allow_html=True)
    st.text("")
    radio = st.sidebar.radio("Choose upload option", ('Upload CSV of stimulus details','Upload PNG image of stimulus'), key='5')
    if radio == 'Upload CSV of stimulus details':
        st.markdown("<h2 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Upload CSV file of stimulus details</strong></h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:normal'>Upload your stimulus and specify its details according to the <strong>CSV template provided</strong>. Alternatively, you can upload your stimulus idea as an image by switching to this option.</p>", unsafe_allow_html=True)
        st.text("")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key='3')
        if uploaded_file is not None:
            st.write("Classifying...")
            new_idea = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            consolidated = consolidate(new_idea, data)
            df_res = matrix_description(consolidated)
            df_res2 = matrix_header(consolidated)
            final_df = one_hot_encode(consolidated, df_res, df_res2)
            final_df = all_together(final_df, df_res, df_res2)
            final_df = predict_uniqueness(final_df)
            final_df = predict_appeal(final_df)
            results_df = pd.concat([final_df['Idea_Name'], final_df['Uniqueness_Prediction'], final_df['Bet_For_Prediction']], axis=1)
            #st.markdown('### Your predicted UNIQUENESS rating is: ' + final_df.iloc[49]['Uniqueness_Prediction'])
            #st.markdown('### Your predicted BET FOR rating is: ' + final_df.iloc[49]['Bet_For_Prediction'])
            st.markdown('### Here are the predictions for your stimulus:')
            st.write("")
            idea = results_df[49:]
            st.write(idea)
    if radio == 'Upload PNG image of stimulus':
        st.markdown("<h2 style='font-family:'IBM Plex Sans',sans-serif;font-weight:700;font-size:2rem'><strong>Upload PNG image of stimulus</strong></h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-weight:normal'>Upload an image of your stimulus idea in <strong>PNG format</strong> to get its predicted ratings. Alternatively, you can upload your stimulus idea in a CSV file by switching to this option.</p>", unsafe_allow_html=True)
        st.text("")
        uploaded_file = st.file_uploader("Choose a PNG file", type="png", key='4')
