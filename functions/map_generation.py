#!/usr/bin/python3

import cufflinks as cf
import plotly
import chart_studio.plotly 
import plotly.tools 
import plotly.graph_objs as go
from  plotly.offline import plot
import pandas as pd

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

def generate_map(filename, map_name, usa= True, 
                 html = True, show_map = True, 
                 source_file_path = 'map/file_for_map/', 
                 html_file_path = 'map/html/'):

    ''' This function generate an interactive map of the world (and USA)
    with the avreage score of beer per countries and their favored beer's style. 
    Is is assumed it will be used in a file in the root of the project. 
    If it's not the case, see: source_file_path and html_file_path.

    Parameters
    ----------
    filename         (string)  :  Name of the source file containing the necessary dataframe for map generation 
                                  should contain a 'location', 'sytle' and 'avg_computed' column.
    map_name         (string)  :  HTML file's name for the resulting interactive map.
    usa              (boolean) :  Activate the generation of the USA's map.
    html             (boolean) :  Activate the generation of html file.
    show_map         (boolean) :  Show the produced map(s).
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
    second = False
    
    for j, country in enumerate(data['location']):
        if "United States" in country:
            mask[j] = True #Prepare mask for united states only dataframe (united_states)
            if second:
                #get rid of all the occurance of United States except best one
                location_country = location_country.drop(j) 
            else:
                #Delete the State
                location_country['location'][j] = "United States"
            second = True
            
    if usa:  #Only activate if we want an USA's view      
        united_states = data[mask]
        for k, state in enumerate(united_states['location']):
            #Only keep the states 
            united_states['location'][k] = state.split('States, ',1)[1]
    #Plot the worldwide figure
    fig_world = go.Figure(data = go.Choropleth(
        locations = location_country['location'], #counties's nams are used to place data on the world map
        locationmode= 'country names',
        z = location_country['avg_computed'], #data that describes the choropleth value-to-color mapping
        text = 'Favored style: '+ location_country['style'], #pop-up for each country 
        colorscale = 'Viridis',
        autocolorscale=False,
        reversescale=True,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_tickprefix = 'avreage rating ',
        colorbar_title = 'style of beer<br>per country',
    ))

    fig_world.update_layout(
        title_text='Beer style and avreage score per countries'
    )
    #print the worldwide map
    if show_map :
        fig_world.show()
    #Create an html file of the map for the site 
    if html : 
        fig_world.write_html(html_file_path + map_name +"_country.html")
        
    #Activate if we want USA map
    if usa:
        #switch state by their abbreviation to fit 
        #the locationmode 'USA-states' of the plotly library
        united_states.location = united_states.location.map(STATES)
        #plot the usa map
        fig_usa = go.Figure(data = go.Choropleth(
            locations = united_states['location'], #abbreviation are used to place data on the USA map
            locationmode= 'USA-states',
            z = united_states['avg_computed'], #data that describes the choropleth value-to-color mapping
            text = 'Favored style: '+ united_states['style'], #pop-up for each country 
            colorscale = 'Viridis',
            autocolorscale=False,
            reversescale=True,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_tickprefix = 'avreage rating ',
            colorbar_title = 'style of beer<br>and avreage ranking<br>in USA',
        ))

        fig_usa.update_layout(
            title_text='Beer style and avreage score per countries',
            geo=dict( scope='usa') #switch from world-map to USA
        )
        #print the map
        if show_map :
            fig_usa.show()
        #generate html file of USA map
        if html :
            fig_usa.write_html(html_file_path + map_name +"_usa.html")
    return

    
