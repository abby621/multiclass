import sys, os
sys.path.append('/Users/Abby/Documents/repos/ei_code/django_ei/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'
from django.conf import settings

import django
django.setup()

from core.models import *
import csv

from math import ceil

import numpy as np

f = open('/Users/abby/Documents/Exchange Initiative/ActivePropertyList.txt','rU')
reader = csv.reader(f, delimiter='\t')
next(reader)
hotel_dict = {rows[0]:{'name':rows[1],'property_category':rows[5],'star_rating':rows[6],'chain_code':rows[7]} for rows in reader}

train_file = open('/Users/abby/Documents/Exchange Initiative/tripletloss/test.txt','rU')
reader2 = csv.reader(train_file,delimiter='\t')
training_ims = list(reader2)

large_cities_file = open('/Users/abby/Documents/Exchange Initiative/largeAmericanCities.csv','rU')
reader3 = csv.reader(large_cities_file,delimiter=',')
large_cities = list(reader3)
large_cities_np = np.empty((len(large_cities),2))
for ix in range(0,len(large_cities)):
    large_cities_np[ix,:] = (float(large_cities[ix][1]),float(large_cities[ix][2]))

def in_city(hotel_loc):
    dlat = np.radians(large_cities_np[:,0]) - np.radians(hotel_loc[0])
    dlon = np.radians(large_cities_np[:,1]) - np.radians(hotel_loc[1])
    a = np.sin(dlat/2.0)**2 + np.cos(hotel_loc[0]) * np.cos(large_cities_np[:,0]) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    if min(km) <= 25:
        return 1
    else:
        return 0

new_train_file = open('/Users/abby/Documents/Exchange Initiative/tripletloss/test_multilabel.txt','a')
expedia_hotels = {}
for t in training_ims:
    traffickCamId = int(t[1])
    # im = Image.objects.filter(path__icontains=t[0].split('/')[-1].split('_',1)[1])[0]
    if traffickCamId not in expedia_hotels:
        hotel = Hotel.objects.get(id=traffickCamId)
        expediaId = hotel.expedia_id
        if expediaId:
            expedia_hotels[traffickCamId] = {}
            expedia_hotels[traffickCamId]['hotelId'] = expediaId
            expedia_hotels[traffickCamId]['hotelLoc'] = (hotel.lat, hotel.lng)
    if traffickCamId in expedia_hotels and expedia_hotels[traffickCamId]['hotelId'] in hotel_dict:
        thisHotel = hotel_dict[expedia_hotels[traffickCamId]['hotelId']]
        if 'suite' in thisHotel['name']:
            is_suite = 1
        else:
            is_suite = 0
        is_urban = in_city(expedia_hotels[traffickCamId]['hotelLoc'])
        # print im.path, is_urban
        property_category = thisHotel['property_category']
        if not property_category or int(thisHotel['property_category']) == 1:
            is_hotel = 1 # property_category for 'hotel'
        else:
            is_hotel = 0
        star_rating = thisHotel['star_rating']
        one_star = 0
        two_star = 0
        three_star = 0
        four_star = 0
        if not star_rating or int(ceil(float(star_rating))) == 3:
            three_star = 1
        elif int(ceil(float(star_rating))) == 1 or int(ceil(float(star_rating))) == 0:
            one_star = 1
        elif int(ceil(float(star_rating))) == 2:
            two_star = 1
        else:
            four_star = 1
        new_train_file.write('%s %s %s %s %s %s %s %s\n' % (t[0], is_suite, is_hotel, is_urban, one_star, two_star, three_star, four_star))
        print 'Including ' + t[0]
    else:
        print 'Not including ' + t[0]

new_train_file.close()
