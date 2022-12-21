#!/usr/bin/python3

import cufflinks as cf
import plotly
import chart_studio.plotly 
import plotly.tools 
import plotly.graph_objs as go
import plotly.express as px
from  plotly.offline import plot
import pandas as pd
import numpy as np
import openai
from sklearn.manifold import TSNE
from PIL import Image
import kaleido
from io import BytesIO
import base64
import os 
STATES = {
"Alabama": "AL",
"Alaska": "AK",
"Arizona": "AZ",
"Arkansas": "AR",
"California": "CA",
"Colorado": "CO",
"Connecticut": "CT",
"Delaware": "DE",
"Florida": "FL",
"Georgia": "GA",
"Hawaii": "HI",
"Idaho": "ID",
"Illinois": "IL",
"Indiana": "IN",
"Iowa": "IA",
"Kansas": "KS",
"Kentucky": "KY",
"Louisiana": "LA",
"Maine": "ME",
"Maryland": "MD",
"Massachusetts": "MA",
"Michigan": "MI",
"Minnesota": "MN",
"Mississippi": "MS",
"Missouri": "MO",
"Montana": "MT",
"Nebraska": "NE",
"Nevada": "NV",
"New Hampshire": "NH",
"New Jersey": "NJ",
"New Mexico": "NM",
"New York": "NY",
"North Carolina": "NC",
"North Dakota": "ND",
"Ohio": "OH",
"Oklahoma": "OK",
"Oregon": "OR",
"Pennsylvania": "PA",
"Rhode Island": "RI",
"South Carolina": "SC",
"South Dakota": "SD",
"Tennessee": "TN",
"Texas": "TX",
"Utah": "UT",
"Vermont": "VT",
"Virginia": "VA",
"Washington": "WA",
"West Virginia": "WV",
"Wisconsin": "WI",
"Wyoming": "WY",
"District of Columbia": "DC",
"American Samoa": "AS",
"Guam": "GU",
"Northern Mariana Islands": "MP",
"Puerto Rico": "PR",
"United States Minor Outlying Islands": "UM",
"U.S. Virgin Islands": "VI",
}


def generate_map(filename, map_name, usa= False, 
                 html = True, show_map = True,
                 title = '',
                 source_file_path = 'map/file_for_map/', 
                 html_file_path = 'map/html/'):

    ''' Author: Gabriel Benato @HOTCHOCOLATE, ADA2022
    This function generate an interactive map of the world (and USA)
    with the average score of beer per countries and their favored beer's style. 
    Is is assumed it will be used in a file at the root of the project. 
    If it's not the case see: source_file_path and html_file_path.

    Parameters
    ----------
    filename         (string)  :  Name of the source file containing the necessary dataframe for map generation 
                                  should contain a 'location', 'style', 'normalized_rating', 'pos_words' and 'neg_words' column.
    map_name         (string)  :  HTML file's name for the resulting interactive map.
    usa              (boolean) :  Activate the generation of the USA's map.
    html             (boolean) :  Activate the generation of html file.
    show_map         (boolean) :  Show the produced map(s).
    title            (string)  :  Title of the plot.
    source_file_path (string)  :  Path to the source file.
    html_file_path   (string)  :  Path to the futur html file.


    Returns
    -------
    None, but may generate html file of interactive map and show generated map.
    '''

    data = pd.read_csv(source_file_path + filename)
    
    ### NOTE: Clean corrrupted data but we should do it before 
    for i, e in enumerate(data['location']):
        if "http" in e or "<" in e:
            data = data.drop(i)
    data.reset_index(inplace = True, drop = True) #reset index so we don't make error due to assumption of continuous index

    # We have multiple occurence of the USA (multiple states) 
    # but we will only keep the best one for the world map 
    # We will also set-up a way to have a look only in the United Sates 
    location_country = data.copy()
    mask = [False] * data.shape[0]
    
    for j, country in enumerate(data['location']):
        if "United States" in country:
            mask[j] = True #Prepare mask for united states only dataframe (united_states)
            if "California" in country:
                #Delete the State
                location_country['location'][j] = "United States"
            else:
                #get rid of all the occurance of United States except Florida since it is the most populated state
                location_country = location_country.drop(j) 
                
    hover_data_world = np.stack((location_country["beer_name"],
                                 location_country["brewery_name"],
                                 location_country["normalized_rating"],
                                 location_country["style"],
                                 location_country["pos_words"],
                                 location_country["neg_words"]), axis=-1)

    #Plot the worldwide figure
    fig_world = go.Figure(data = go.Choropleth(
        locations = location_country['location'], #counties's nams are used to place data on the world map
        locationmode= 'country names',
        z = location_country['normalized_rating'], #data that describes the choropleth value-to-color mapping
        text = location_country['location'], #pop-up for each country 
        colorscale = 'Viridis',
        autocolorscale=False,
        reversescale=True,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_tickprefix = 'average rating ',
        colorbar_title = "Mean average rating",
        customdata = hover_data_world,
        hovertemplate="""   <br><b>Country</b>: %{text}
                            <br><b>Beer</b>: %{customdata[0]}
                            <br><b>Brewery</b>: %{customdata[1]}
                            <br><b>Mean Rating</b>: %{customdata[2]:.2f}
                            <br><b>Type</b>: %{customdata[3]}
                            <br><b>Positive words in reviews</b>: %{customdata[4]}
                            <br><b>Positive words in reviews</b>: %{customdata[5]}<br><extra></extra>"""
    ))

    fig_world.update_layout(
        title_text=title,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', 
    )
    #print the worldwide map
    if show_map :
        fig_world.show()
    #Create an html file of the map for the site 
    if html : 
        fig_world.write_html(html_file_path + map_name +"_country.html")
        
    #Activate if we want USA map
    if usa:
        united_states = data[mask]
        for k, state in enumerate(united_states['location']):
            #Only keep the states 
            united_states['location'][k] = state.split('States, ',1)[1]
            
        #switch state by their abbreviation to fit 
        #the locationmode 'USA-states' of the plotly library
        united_states.location = united_states.location.map(STATES)
        hover_data_usa = np.stack((united_states["beer_name"],
                                 united_states["brewery_name"],
                                 united_states["normalized_rating"],
                                 united_states["style"],
                                 united_states["pos_words"],
                                 united_states["neg_words"]), axis=-1)
        
        #plot the usa map
        fig_usa = go.Figure(data = go.Choropleth(
            locations = united_states['location'], #abbreviation are used to place data on the USA map
            locationmode= 'USA-states',
            z = united_states['normalized_rating'], #data that describes the choropleth value-to-color mapping
            text = location_country['location'], #pop-up for each country 
            colorscale = 'Viridis',
            autocolorscale=False,
            reversescale=True,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_tickprefix = 'average rating ',
            colorbar_title = 'Mean average rating',
            customdata = hover_data_usa,
            hovertemplate="""   <br><b>States</b>: %{text}
                            <br><b>Beer</b>: %{customdata[0]}
                            <br><b>Brewery</b>: %{customdata[1]}
                            <br><b>Mean Rating</b>: %{customdata[2]:.2f}
                            <br><b>Type</b>: %{customdata[3]}
                            <br><b>Positive words in reviews</b>: %{customdata[4]}
                            <br><b>Positive words in reviews</b>: %{customdata[5]}<br><extra></extra>"""
    

    
        ))

        fig_usa.update_layout(
            title_text='Zoom on the USA',
            geo=dict( scope='usa'), #switch from world-map to USA
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', 
        )
        #print the map
        if show_map :
            fig_usa.show()
        #generate html file of USA map
        if html :
            fig_usa.write_html(html_file_path + map_name +"_usa.html")
    return

def combine_neg_pos_and_favoured_beer(neg_pos_filename, favoured_filename, combined_filename, 
                                       source_file_path = 'map/file_for_map/'):
    ''' 
    Function combines dataset from positive-negative word analysis and favoured beer analysis.

    Parameters
    ----------    
    neg_pos_filename  (string)  :  Name of the source file from the favoured beer analysis.
    favoured_filename (string)  :  Name of the source file from the favoured beer analysis.
    combined_filename (string)  :  Name of the resulting combined file.
    source_file_path  (string)  :  Path to the source files
    '''
    neg_pos = pd.read_csv(source_file_path + neg_pos_filename)
    favoured_beer = pd.read_csv(source_file_path + favoured_filename)


    merged_data = favoured_beer.merge(neg_pos[["location","neg_words","pos_words"]],
                                               how="outer",left_on="location",right_on="location")

    merged_data['pos_words'] = merged_data['pos_words'].fillna('Unknown')
    merged_data['neg_words'] = merged_data['neg_words'].fillna('Unknown')
    merged_data.to_csv(source_file_path + combined_filename)
    return

#All functions used in this notebook
def find_high_rating_review(df):
    """ To be used as an aggregation function of GroupBy object (e.g.  pandas.DataFrame.groupby(...).agg(find_high_rating_review))
    Filters the input dataframe by selecting rows with reviews that verify:
    1. Max rating was given
    2. The overall rating (sum of taste, aroma, ...) is not 5 points less than the maximum value
    3. The review is not empty (has at least 2 characters)
    """
    max_rating = df["rating"].max()
    max_overall = df["overall"].max()
    nice_reviews = df[(df["rating"] == max_rating) & (df["overall"] >  max_overall - 5) & (len(df["text"]) > 1)]["text"]
    nice_reviews_sample = nice_reviews.sample(n=min(5,len(nice_reviews)))
    concatenated_nice_reviews = nice_reviews_sample.str.cat(sep=' ')
    return pd.DataFrame.from_dict([{"beer_id" : df["beer_id"].iloc[0], "beer_name" : df["beer_name"].iloc[0], "style": df["style"].iloc[0],"brewery_name": df["brewery_name"].iloc[0], "good_reviews" : concatenated_nice_reviews}])


def retrieve_reviews_SAT(website):
    """ Retrieves reviews corresponding to beers sold on SAT in the dataset corresponding to a given 'website' and to beers favoured by each country
    Parameters
    ----------
    website     (string)  : name of the dataset. Either 'RateBeer' or 'BeerAdvocate
    Returns
    -------
    (series) best reviews of the beers favoured by each country
    (series) best reviews of the beers sold at SAT that are found in 'website'
    
    
    """
    if website == "RateBeer":
        TOTAL_CSV = 72
        acronym = "RB"
        fav_beer = pd.read_csv("DATA/favourite_beer_RB.csv")["beer_id"].unique().tolist()
    if website == "BeerAdvocate":
        TOTAL_CSV = 26
        acronym = "BA"
        fav_beer = pd.read_csv("DATA/favourite_beer_BA.csv")["beer_id"].unique().tolist()

    #Retrieve favourite beer of each country according to users for RB
    temp = pd.read_csv(f"DATA/{website}_reviews_part_0.csv")
    best_reviews = temp[temp["beer_id"].isin(fav_beer)]
    
    ##Retrieve reviews for SAT beers
    SAT_RB_best_beers = pd.read_csv(f"DATA/predicted_SAT_{acronym}_sorted.csv",index_col=0)
    #Take 10 best SAT beers according to users
    bestSAT_RB = SAT_RB_best_beers[f"{acronym}_beer_id"].values[0:13]
    best_reviews_SAT = temp[temp["beer_id"].isin(bestSAT_RB)]
    for i in range(1,TOTAL_CSV):
        #Iterate over all the partitioned dataset and populate a dataframe only with the reviews of SAT beers
        temp = pd.read_csv(f"DATA/{website}_reviews_part_{i}.csv")
        best_reviews = pd.concat([best_reviews,temp[temp["beer_id"].isin(fav_beer)]],join="outer")
        best_reviews_SAT = pd.concat([best_reviews_SAT,temp[temp["beer_id"].isin(bestSAT_RB)]],join="outer")
    best_reviews = best_reviews.groupby(by="beer_id").apply(find_high_rating_review).reset_index(drop=True)
    best_reviews_SAT = best_reviews_SAT.groupby(by="beer_id").apply(find_high_rating_review).reset_index(drop=True)
    return best_reviews, best_reviews_SAT


def add_country_column(target_df,country_csv_path,california_as_usa=True):
    """
    Given a dataset of beer reviews 'target_df', augments the dataset with the location of the reviewer from an accessory dataset located in 'country_csv_path'
    """
    #Recover the dataframe of favourite beer for users of each country. 
    #Drop countries for which there were no enough reviewers (tagged as favourite beer_id = -1)
    countries = pd.read_csv(country_csv_path)
    countries = countries[~ (countries["beer_id"] == -1.)]
    best_with_countries = target_df.merge(countries[["beer_id","location"]],how="left",on="beer_id")
    #Clean individual states of United States, by keeping only California
    if california_as_usa:
        mask = (best_with_countries["location"].str.contains("United States")) & ~(best_with_countries["location"].str.contains("California"))
        best_with_countries = best_with_countries[~mask]
    return best_with_countries


def request_embeddings(series,verbose=True):
    """
    Given a series of textual reviews, sends a series of calls to OpenAI Embeddings API for the ADA model embeddings. 
    Calls are sent on 6 seconds interval to avoid RateLimitError from the OpenAI API.
    
    Parameters
    ----------
    series     (pd.Series)  : Series of reviews to be embedded by ADA-002
    verbose    (bool)       : if True, reviews are printed as they are sent to the OpenAI API
    Returns
    -------
    (numpy.array) (len(series),1956) length numpy array corresponding to OpenAi ADA embeddings for each review sent
    
    
    """
    import time
    from tenacity import retry, wait_random_exponential, stop_after_attempt
    #We use exponential backoff to limit adaptively our request rate
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding_exponential_bo(text: str, engine="text-embedding-ada-002") -> list[float]:

        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]
    corpus = series.values
    embeddings = []
    #We send request to the API on a loop in order to control the request rate and avoid being limited
    for (i,x) in enumerate(corpus):
        if verbose:
            print("Sending API request to embed the following review:\n")
            print(x)
        embeddings.append(get_embedding_exponential_bo(x, engine='text-embedding-ada-002'))
        #Toggle the waiting time between requests, max request rate is 20/min
        time.sleep(6)

    embeddings = np.array(embeddings)
    if verbose:
        print(embeddings.shape)
    return embeddings

### Plotting a t-SNE with plotly with custom markers ###

def plot_tsne(SAT_embeddings,dataset_embeddings,sat_df,dataset_df,perplexity=10,country_size=20,beer_size=20,acronym="",title=""):
    """
    Given arrays of embeddings corresponding to reviews of SAT and of the preferred beers of each country for a given dataset,
    plot a t-SNE graph where embeddings corresponding to SAT beers are rendered as beer images and where favourite beers are rendered as the flag of the country that prefers them.
    The plot can be customized for perplexity and marker size.
    """
    #Beers sold on SAT should be shown with their image, while embeddings representing favoured beers for each
    #country should be shown with the country flag.
    tsne_vectors = np.concatenate([SAT_embeddings,dataset_embeddings])
    sat_label = np.ones(len(SAT_embeddings))
    non_sat_label = np.zeros(len(dataset_embeddings))
    label = np.concatenate([sat_label,non_sat_label])
    SAT_ids = sat_df["beer_id"].values
    df = pd.DataFrame()
    hover_data = np.stack((pd.concat([sat_df["beer_name"],dataset_df["beer_name"]]),pd.concat([sat_df["brewery_name"],dataset_df["brewery_name"]]),pd.concat([sat_df["style"],dataset_df["style"]])),axis=-1)
    df["is_sat_beer"] = label
    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=42, init="random", learning_rate=200,n_iter=10000
    )

    sat_beer_id = [str(number) for number in SAT_ids]
    vis_dims2 = tsne.fit_transform(tsne_vectors)
    df["x"] = vis_dims2[:,0]
    df["y"] = vis_dims2[:,1]
    df["general_id"] = sat_beer_id + list(dataset_df["location"].values)
    df["beer"] = list(sat_df["beer_name"].values) + list(dataset_df["beer_name"].values)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_name="general_id",
        hover_data=["beer"],
        labels=dict(x="t-SNE  first axis", y="t-SNE second axis")

    )
    fig.update_traces(marker_color="rgba(0,0,0,0)",mode='markers',
                      customdata=hover_data,  
                      hovertemplate="""
                            <br><b>Beer</b>: %{customdata[0]}
                            <br><b>Brewery</b>: %{customdata[1]}
                            <br><b>Style</b>: %{customdata[2]}<br><extra></extra>""")
    maxDim = df[["x", "y"]].max().idxmax()
    maxi = df[maxDim].max()
    for i, row in df.iterrows():
        general_id = row["general_id"].replace(" ","-")
        if row.is_sat_beer:
            fig.add_layout_image(
                dict(
                    source=Image.open(f"Images/beers/{general_id}.0.png"),
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=row["x"],
                    y=row["y"],
                    sizex=beer_size,
                    sizey=beer_size,
                    sizing="contain",
                    opacity=1,
                    layer="above"
                )
            )
        #We do not have flag images for certain countries ::sad_face::
        elif general_id not in["Afghanistan","Albania","Bahrain","Benin","Bosnia-and-Herzegovina","Botswana","Bulgaria","Burkina-Faso","Burundi","Cambodia","Central-African-Republic","Comoros","Congo,-Dem.-Rep."
                          ,"Congo,-Rep.","Costa-Rica","Cote-d'Ivoire","El-Salvador","Equatorial-Guinea","Eritrea","Ethiopia","Gabon","Gambia","Ghana","Guatemala","Guinea","Guinea-Bissau","Haiti","Honduras","Hong-Kong,-China","Hungary","Indonesia",
                         "Iraq","Jordan","Kenya","Korea,-Dem.-Rep.","Korea,-Rep.","Kuwait","Lebanon","Lesotho","Liberia","Libya","Madagascar",
                         "Malawi","Malaysia","Mali","Mauritania","Mauritius","Mongolia","Montenegro","Mozambique","Myanmar","Namibia","Nepal","Nicaragua","Niger","Nigeria","Oman","Pakistan","Panama","Paraguay",
                         "Philippines","Puerto-Rico","Reunion","Romania","Rwanda","Sao-Tome-and-Principe","Saudi-Arabia","Senegal","Serbia","Sierra-Leone","Singapore","Slovak-Republic","Slovenia","Somalia","Sri-Lanka",
                         "Sudan","Swaziland","Syria","Taiwan","Tanzania","Togo","Trinidad-and-Tobago","Tunisia","Uganda","United-Kingdom","Vietnam","West-Bank-and-Gaza","Yemen,-Rep.","Zambia","Zimbabwe"]:
            fig.add_layout_image(
                dict(
                    source=Image.open(f"Images/country_flags/icons8-{general_id.lower()}-100.png"),
                    xref="x",
                    yref="y",
                    xanchor="center",
                    yanchor="middle",
                    x=row["x"],
                    y=row["y"],
                    sizex=country_size,
                    sizey=country_size,
                    sizing="contain",
                    opacity=1,
                    layer="above"
                )
            )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True,showgrid=False,zeroline=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True,showgrid=False,zeroline=False)                        , 
    fig.update_layout(height=600, 
                      width=1000, 
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title_text=title)

    fig.show()

    fig.write_html(f"Images/{acronym}_tsne.html",config = {'displayModeBar': True})
def retrieve_style_list(website):
    """
    Returns a list of all beer styles of the dataset of given 'website'

    Parameters
    ----------
    website         (string)  :  website corresponding to the dataset being processed. Either 'RateBeer' or 'BeerAdvocate'

    Returns
    -------
    (list) : A lexicographically sorted list of beer styles
    """
    if website == "RateBeer":
        TOTAL_CSV = 72
    if website == "BeerAdvocate":
        TOTAL_CSV = 26        

    style_set = set()
    for index in range(0,TOTAL_CSV):
            temp = pd.read_csv(f"DATA/{website}_reviews_part_{index}.csv",usecols=["text","style"],low_memory=False).astype(str)
            styles = set(temp["style"].unique())
            style_set.update(styles)
    style_list = list(style_set)
    style_list.sort()
    return style_list
def plot_wordcloud_dropdown(website):
    """
    Plots wordclouds corresponding to beer reviews for all beer styles in a given 'website'. Style can be chosen with a dropdown menu
    """
    if website == "RateBeer":
        acronym = "RB"
    if website == "BeerAdvocate":
        acronym = "BA"  
    # Load img
    img_list = os.listdir("Images/word_clouds")
    # Initialize figure
    fig = go.Figure(layout=go.Layout(width=500, height=500,
                                    xaxis=dict(range=[140, 430],
                                            fixedrange = True),
                                    yaxis=dict(range=[400, 50],
                                            fixedrange = True
                                    ),
                                    ))
    # image dimensions (pixels)
    # Generate an image starting from a numerical function
    # Add Traces
    style_list = retrieve_style_list(website)
    for i,style in enumerate(style_list):
        if "/" in style:
            style = style.replace("/","")
        
        pil_img = Image.open(f'Images/word_clouds/{acronym}_{style}_wordcloud.png') # PIL image object
        prefix = "data:image/png;base64,"
        with BytesIO() as stream:
            pil_img.save(stream, format="png")
            base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
        if i == 0:
            goImg = go.Image(source=base64_string,
                            x0=0, 
                            y0=0,
                            dx=0.5,
                            dy=0.5,
                            visible = True,)
        else:
            goImg = go.Image(source=base64_string,
                        x0=0, 
                        y0=0,
                        dx=0.5,
                        dy=0.5,
                        visible = False,)
        fig.add_trace(goImg)
        fig.update_traces(
                    hovertemplate = None,
                    hoverinfo = "skip")
    mask_list = []
    mask = np.arange(0,len(img_list))
    for i in range(len(img_list)):
        mask_list.append(list(mask==i))
    buttons = [{'label': style, 'method':'update','args':[{"visible":mask_list[i]}]} for i,style in enumerate(style_list)]
    fig.update_layout(template="simple_white",
                updatemenus=[dict(
                active=1,
                x=0.8,
                y=1.1,
                buttons=buttons,
            )
        ])
    # Add Annotations and Buttons
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    # Set title
    fig.show()
    fig.write_html(f"Images/{acronym}_wordclouds.html",config = {'displayModeBar': False})