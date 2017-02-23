import csv
import numpy as np
import shutil
import random

old_file_path = '/project/focus/abby/multiclass/datasets/star_rating/train.txt'
old_file = open(old_file_path,'rU')
reader = csv.reader(old_file,delimiter=' ')
imList = list(reader)

starRatings = [int(im[1]) for im in imList]
npStarRatings = np.asarray(starRatings)

oneAndTwoStar = np.where(npStarRatings==0)[0]
threeStar = np.where(npStarRatings==1)[0]
fourStar = np.where(npStarRatings==2)[0]

maxNum = np.max((oneAndTwoStar.shape[0],threeStar.shape[0],fourStar.shape[0]))

new_file_path = old_file_path.split('.')[0] + '_oversampled.txt'
shutil.copy(old_file_path,new_file_path)
new_file = open(new_file_path,'a')

numOneTwoStar = oneAndTwoStar.shape[0]
while numOneTwoStar < maxNum:
    print numOneTwoStar
    toAdd = imList[random.choice(oneAndTwoStar)]
    new_file.write('%s %s\n' (toAdd[0], toAdd[1]))
    numOneTwoStar += 1

numThreeStar = threeStar.shape[0]
while numThreeStar < maxNum:
    print numThreeStar
    toAdd = imList[random.choice(threeStar)]
    new_file.write('%s %s\n' (toAdd[0], toAdd[1]))
    numThreeStar += 1

numFourStar = fourStar.shape[0]
while numFourStar < maxNum:
    print numFourStar
    toAdd = imList[random.choice(fourStar)]
    new_file.write('%s %s\n' (toAdd[0], toAdd[1]))
    numFourStar += 1
