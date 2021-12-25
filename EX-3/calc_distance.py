import sys
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from xml.dom.minidom import parse

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    from xml.dom.minidom import getDOMImplementation
    def append_trkpt(pt, trkseg, doc):
        trkpt = doc.createElement('trkpt')
        trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
        trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
        trkseg.appendChild(trkpt)
    
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def map_func(data): # referred to https://www.youtube.com/watch?v=ThYmvjILRKk&t=1908s
    lat = float(data.getAttribute('lat'))
    lon = float(data.getAttribute('lon'))
    return lat, lon

def haversine (data): #referred to https://www.movable-type.co.uk/scripts/latlong.html
    R = 6371000
    lat1 = np.radians(data.lat)
    lat2 = np.radians(data.next_lat)
    delta_lat = np.radians(data.next_lat - data.lat)
    delta_lon = np.radians(data.next_lon - data.lon)    
    a = np.sin(delta_lat/2) **2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) 
    d = R * c
    return d
    
def distance(df):    
    df['next_lat'] = df.lat.shift(periods=-1)
    df['next_lon'] = df.lon.shift(periods=-1)    
    dist = df.apply(haversine, axis = 1)    
    return np.sum(dist)
    
def read_gpx (gpxFile):
    parse_result = parse(gpxFile)
    trkpt = parse_result.getElementsByTagName('trkpt')
    df = pd.DataFrame(list(map(map_func, trkpt)),columns=['lat', 'lon'])# referred to https://www.youtube.com/watch?v=ThYmvjILRKk&t=1908s
    return df

def smooth(df):
    kalman_data = df[['lat', 'lon']] * 100000

    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([20, 20]) ** 2 #200
    transition_covariance = np.diag([10, 10]) ** 2 #0.00001
    transition = [[1, 0], [0, 1]] 

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition
    )
    
    kalman_smoothed, state_cov = kf.smooth(kalman_data)   
    newDF = pd.DataFrame(kalman_smoothed, columns=['lat', 'lon']) / 100000
    return newDF

def main():
    points = read_gpx(sys.argv[1])
    print('Unfiltered distance: %0.2f' % (distance(points)))
    
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % (distance(smoothed_points)))
    
    output_gpx(smoothed_points, 'out.gpx')
    
if __name__ == '__main__':
    main()
