import cv2
import numpy as np;
from enum import Enum
import math
from scipy import stats
from matplotlib import pyplot as plt

image_name = "setgame1.jpg"
threshold_name = "setgame1-threshold.png"
card_contour_name = "setgame-boxes.png"
training_path = "./Training_Images/"


class Shape(Enum):
	UNKNOWN = 0
	DIAMOND = 1
	ELLIPSE = 2
	SQUIGGLE = 3

class Color(Enum):
	UNKNOWN = 0
	GREEN = 1
	RED = 2
	PURPLE = 3

class Fill(Enum):
	UNKNOWN = 0
	NONE = 1
	SOLID = 2
	STRIPED = 3

class Number(Enum):
	UNKNOWN = 0
	ONE = 1
	TWO= 2
	THREE = 3

class Card:

    def __init__(self):
        self.contour = [] 
        self.width, self.height = 0, 0 
        self.center = [] 
        self.shape = Shape.UNKNOWN
        self.color = Color.UNKNOWN
        self.fill = Fill.UNKNOWN
        self.number = Number.UNKNOWN

class Training_card:

	def __init__(self):
		self.image = []
		self.shape = Shape.UNKNOWN
		self.color = Color.UNKNOWN
		self.fill = Fill.UNKNOWN
		self.number = Number.UNKNOWN

	def prettyprint(self):
		print(self.shape)
		print(self.color)
		print(self.fill)
		print(self.number)

def parse_number(string):
	if(string == "One"):
		return Number.ONE
	if(string == "Two"):
		return Number.TWO
	if(string == "Three"):
		return Number.THREE
	return Number.UNKNOWN

def parse_fill(string):
	if(string == "None"):
		return Fill.NONE
	if(string == "Solid"):
		return Fill.SOLID
	if(string == "Striped"):
		return Fill.STRIPED
	return Fill.UNKNOWN

def parse_color(string):
	if(string == "Green"):
		return Color.GREEN
	if(string == "Red"):
		return Color.RED
	if(string == "Purple"):
		return Color.PURPLE
	return Color.UNKNOWN

def parse_shape(string):
	if(string == "Diamond"):
		return Shape.DIAMOND
	if(string == "Ellipse"):
		return Shape.ELLIPSE
	if(string == "Squiggle"):
		return Shape.SQUIGGLE
	return Shape.UNKNOWN


def initialize_training_cards(filepath):
	# Loads in all training images. These images will be compared with the live data 
	# to obtain the type of card
	# I am hoping that if the aspect ratio is the same everything will work out. 
	# If the camera angle is different this might now work. 

	training_cards = []
	CARD_NAMES = ["One_None_Purple_Ellipse","One_Striped_Green_Ellipse","One_Solid_Green_Ellipse", \
	              "Three_None_Purple_Diamond","Three_None_Purple_Ellipse","Three_Striped_Red_Squiggle",\
	              "Three_Solid_Purple_Diamond","Three_Solid_Purple_Ellipse","Three_Solid_Red_Squiggle",\
	              "Two_None_Green_Diamond","Two_Solid_Red_Diamond","Two_Striped_Red_Diamond"]
	for name in CARD_NAMES:
		training_card = Training_card()
		training_card.image = cv2.imread(filepath+name+".png")
		name = name.split("_")
		training_card.number = parse_number(name[0])
		training_card.fill   = parse_fill(name[1])
		training_card.color  = parse_color(name[2])
		training_card.shape  = parse_shape(name[3])
		training_cards.append(training_card)

	return training_cards;

def isolate_cards(image,threshold_name,card_contour_name):
	# Constants for thresholding
	THRESHOLD_LEVEL = 150
	MAX_BINARY_VALUE = 255

	# To remove random rectangular contours in the background 
	MIN_CARD_SIZE = 2500

	# Read in image, grayscale, and apply blur
	im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	im_blur = cv2.GaussianBlur(im_gray,(5,5),0)

	# Apply threshold to image and save
	retval, im_threshold = cv2.threshold(im_blur,THRESHOLD_LEVEL,MAX_BINARY_VALUE,cv2.THRESH_BINARY)
	cv2.imwrite(threshold_name, im_threshold)

	# Extract external contours from image
	src_image,contours,hierarchy = cv2.findContours(im_threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	contour_list = []

	# Find the contours that have four corners and an area > 100	 
	num_cards = 0
	for contour in contours:
		epsilon = 0.1*cv2.arcLength(contour,True)
		approx = cv2.approxPolyDP(contour,epsilon,True)
		if ((len(approx) == 4) and (cv2.contourArea(contour) > MIN_CARD_SIZE)):
			num_cards = num_cards + 1;
			contour_list.append(contour)
			image = cv2.drawContours(image,[contour],-1,(0,255,0),10)

	print("Number of cards detected: ",len(contour_list))

	cv2.imwrite(card_contour_name, image)

	return contour_list


def shape_type(contour_list):

	ratio_list = []
	for contour in contour_list:
		x,y,w,h = cv2.boundingRect(contour)
		ratio_list.append(cv2.contourArea(contour_list[0])/(w*h))

	print("Ratio list:", ratio_list)

	mean_ratio = np.mean(ratio_list)

	if(mean_ratio > .8):
		return Shape.ELLIPSE
	elif(mean_ratio >.6):
		return Shape.SQUIGGLE
	else:
		return Shape.DIAMOND
	return Shape.UNKNOWN

def shape_fill(contours,hierarchy,contour_indexes):
	MIN_SHAPE_SIZE = 100

	children_count = []
	for contour_index in contour_indexes:
		count = 0
		for i in range(len(hierarchy[0])):
			if(hierarchy[0][i][3] == contour_index) and (cv2.contourArea(contours[i]) > 100):
				count = count +1
		children_count.append(count)

	print("Children count:", children_count)

	mean = np.mean(children_count)

	if(mean < 1):
		return Fill.SOLID
	elif (mean < 3):
		return Fill.NONE
	else:
		return Fill.STRIPED
	return Fill.UNKNOWN

def shape_color(image,contour):
	print("nothing")

		
def parse_card(image,card_contour,index):
	#Minimum Shape size
	MIN_SHAPE_SIZE = 1000

	# Constants for thresholding
	THRESHOLD_LEVEL = 175
	MAX_BINARY_VALUE = 255

	# Get size of rectangle
	x,y,w,h = cv2.boundingRect(card_contour)
	
	# Get grayscale of card
	im_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	card_gray = im_gray[y:y+h,x:x+w]
	card_color = image[y:y+h,x:x+w]

	cv2.imwrite("card"+index+".png", card_color)
	
	# Create mask
	mask = np.zeros(card_gray.shape,np.uint8)
	shifted_contour = card_contour - (x,y)
	cv2.drawContours(mask,[shifted_contour],-1,1,-1)
	
	# Apply mask and threshold
	extracted = 255 - mask*(255-card_gray)
	retval, extracted_thresh = cv2.threshold(extracted,THRESHOLD_LEVEL,255,cv2.THRESH_BINARY)

	# Save threshold card
	cv2.imwrite("card_threshold"+index+".png", extracted_thresh)

	# Get contours
	inverted = (255-extracted_thresh)
	src_image,contours,hierarchy = cv2.findContours(inverted,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	# save contours
	# output = cv2.drawContours(card_color, contours, -1, (0,255,0), 3)
	# cv2.imwrite("card_contors.png", output)

	MIN_SHAPE_SIZE = .05*w*h
	MAX_SHAPE_SIZE = .9*w*h
	
	contour_indexes = []
	contour_list = []
	for i in range(len(contours)):
		if ((cv2.contourArea(contours[i]) > MIN_SHAPE_SIZE) and 
			(cv2.contourArea(contours[i]) > MIN_SHAPE_SIZE) and
			(hierarchy[0][i][3] == -1)):
			contour_indexes.append(i)
			contour_list.append(contours[i])


	if(len(contour_list) == 1):
		number = Number.ONE
	elif(len(contour_list) == 2):
		number = Number.TWO
	elif(len(contour_list) == 3):
		number = Number.THREE
	else:
		number = Number.UNKNOWN

	fill = shape_fill(contours,hierarchy,contour_indexes)

	shape = shape_type(contour_list)

	print("Shapes: ", len(contour_list))
	print(fill)
	print(shape)

	# Get size of rectangle
	x,y,w,h = cv2.boundingRect(contour_list[0])
	shape_color = card_color[y:y+h,x:x+w]
	
	# Create mask
	mask = np.zeros(shape_color.shape,np.uint8)
	shifted_contour = contour_list[0] - (x,y)
	cv2.drawContours(mask,[shifted_contour],-1,(1,1,1),-1)
	
	# Apply mask
	extracted = mask*shape_color

	boundaries = [
	([15, 15, 150], [100, 75, 225]),
	([15, 90, 15], [100, 225, 100]),
	([50, 15, 50], [150, 90, 150])]


	color_sums = []
	for i, (lower, upper) in enumerate(boundaries):
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
	 
		# find the colors within the specified boundaries and apply
		# the mask
		mask = cv2.inRange(extracted, lower, upper)
		output = cv2.bitwise_and(extracted, extracted, mask = mask)

		color_sums.append(np.sum(output))
	 
		# # show the images
		# cv2.imshow("images", np.hstack([extracted, output]))
		# cv2.waitKey(300)

	if(color_sums[0] > color_sums[1] and color_sums[0] > color_sums[2]):
		color = Color.RED
	elif(color_sums[1] > color_sums[0] and color_sums[1] > color_sums[2]):
		color = Color.GREEN
	else:
		color = Color.PURPLE

	print(color)
	cv2.imwrite("card_shape"+index+".png", extracted)
	print("Card number: ", str(index))


	font                   = cv2.FONT_HERSHEY_SIMPLEX
	fontScale              = 1
	fontColor              = (247, 183, 51)
	lineType               = 3

	x,y,w,h = cv2.boundingRect(card_contour)		
	image = cv2.imread(card_contour_name)

	text = shape.name
	topLeftCornerOfText    = (x+50,y+50)
	cv2.putText(image,text, 
		topLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType)

	text = number.name
	topLeftCornerOfText    = (x+50,y+100)
	cv2.putText(image,text, 
		topLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType)

	text = color.name
	topLeftCornerOfText    = (x+50,y+150)
	cv2.putText(image,text, 
		topLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType)


	text = fill.name
	topLeftCornerOfText    = (x+50,y+200)
	cv2.putText(image,text, 
		topLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType)

	text = "Card number: " + str(index)
	topLeftCornerOfText    = (x+50,y+250)
	cv2.putText(image,text, 
		topLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType)

	cv2.imwrite(card_contour_name, image)



# training_cards = initialize_training_cards(training_path)
image = cv2.imread(image_name)
contour_list = isolate_cards(image, threshold_name, card_contour_name)

image = cv2.imread(image_name)
parse_card(image,contour_list[11],str(11))

image = cv2.imread(image_name)
for i in range(len(contour_list)):
	parse_card(image,contour_list[i],str(i))








