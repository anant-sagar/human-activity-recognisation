import streamlit as st
import os
import cv2
import numpy as np
import argparse
import imutils
from PIL import Image
from ha_reconition import execute

st.image('img/img1.jpg')
st.title("Human Activity Recognition using AI",)
st.success('Our human activity recognition model can recognize over 400 activities with 78.4-94.5% accuracy (depending on the task), see that list of activities below')

st.markdown('''
    <style>
        section.main{
            	background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
	            background-size: 400% 400%;
	            animation: gradient 7s ease infinite;
        }
        @keyframes gradient {
            0% {
                background-position: 0% 50%;
            }
            50% {
                background-position: 100% 50%;
            }
            100% {
                background-position: 0% 50%;
            }
        }
    </style>
''',unsafe_allow_html=True)

x = st.checkbox('view description')
if x:
    st.header("Description")
    st.write('''
    Some of the practical applications of human activity recognition include:

    - Automatically classifying/categorizing a dataset of videos on disk.
    - Training and monitoring a new employee to correctly perform a task (ex., proper steps and procedures when making a pizza, including rolling out the dough, heating oven, putting on sauce, cheese, toppings, etc.).
    - Verifying that a food service worker has washed their hands after visiting the restroom or handling food that could cause cross-contamination (i.e,. chicken and salmonella).
    - Monitoring bar/restaurant patrons and ensuring they are not over-served.
    ''')
o = st.checkbox('list of recognizable activityies')

if o:
    st.markdown('''
- abseiling
- air drumming
- answering questions
- applauding
- applying cream
- archery
- arm wrestling
- arranging flowers
- assembling computer
- auctioning
- baby waking up
- baking cookies
- balloon blowing
- bandaging
- barbequing
- bartending
- beatboxing
- bee keeping
- belly dancing
- bench pressing
- bending back
- bending metal
- biking through snow
- blasting sand
- blowing glass
- blowing leaves
- blowing nose
- blowing out candles
- bobsledding
- bookbinding
- bouncing on trampoline
- bowling
- braiding hair
- breading or breadcrumbing
- breakdancing
- brush painting
- brushing hair
- brushing teeth
- building cabinet
- building shed
- bungee jumping
- busking
- canoeing or kayaking
- capoeira
- carrying baby
- cartwheeling
- carving pumpkin
- catching fish
- catching or throwing baseball
- catching or throwing frisbee
- catching or throwing softball
- celebrating
- changing oil
- changing wheel
- checking tires
- cheerleading
- chopping wood
- clapping
- clay pottery making
- clean and jerk
- cleaning floor
- cleaning gutters
- cleaning pool
- cleaning shoes
- cleaning toilet
- cleaning windows
- climbing a rope
- climbing ladder
- climbing tree
- contact juggling
- cooking chicken
- cooking egg
- cooking on campfire
- cooking sausages
- counting money
- country line dancing
- cracking neck
- crawling baby
- crossing river
- crying
- curling hair
- cutting nails
- cutting pineapple
- cutting watermelon
- dancing ballet
- dancing charleston
- dancing gangnam style
- dancing macarena
- deadlifting
- decorating the christmas tree
- digging
- dining
- disc golfing
- diving cliff
- dodgeball
- doing aerobics
- doing laundry
- doing nails
- drawing
- dribbling basketball
- drinking
- drinking beer
- drinking shots
- driving car
- driving tractor
- drop kicking
- drumming fingers
- dunking basketball
- dying hair
- eating burger
- eating cake
- eating carrots
- eating chips
- eating doughnuts
- eating hotdog
- eating ice cream
- eating spaghetti
- eating watermelon
- egg hunting
- exercising arm
- exercising with an exercise ball
- extinguishing fire
- faceplanting
- feeding birds
- feeding fish
- feeding goats
- filling eyebrows
- finger snapping
- fixing hair
- flipping pancake
- flying kite
- folding clothes
- folding napkins
- folding paper
- front raises
- frying vegetables
- garbage collecting
- gargling
- getting a haircut
- getting a tattoo
- giving or receiving award
- golf chipping
- golf driving
- golf putting
- grinding meat
- grooming dog
- grooming horse
- gymnastics tumbling
- hammer throw
- headbanging
- headbutting
- high jump
- high kick
- hitting baseball
- hockey stop
- holding snake
- hopscotch
- hoverboarding
- hugging
- hula hooping
- hurdling
- hurling (sport)
- ice climbing
- ice fishing
- ice skating
- ironing
- javelin throw
- jetskiing
- jogging
- juggling balls
- juggling fire
- juggling soccer ball
- jumping into pool
- jumpstyle dancing
- kicking field goal
- kicking soccer ball
- kissing
- kitesurfing
- knitting
- krumping
- laughing
- laying bricks
- long jump
- lunge
- making a cake
- making a sandwich
- making bed
- making jewelry
- making pizza
- making snowman
- making sushi
- making tea
- marching
- massaging back
- massaging feet
- massaging legs
- massaging person's head
- milking cow
- mopping floor
- motorcycling
- moving furniture
- mowing lawn
- news anchoring
- opening bottle
- opening present
- paragliding
- parasailing
- parkour
- passing American football (in game)
- passing American football (not in game)
- peeling apples
- peeling potatoes
- petting animal (not cat)
- petting cat
- picking fruit
- planting trees
- plastering
- playing accordion
- playing badminton
- playing bagpipes
- playing basketball
- playing bass guitar
- playing cards
- playing cello
- playing chess
- playing clarinet
- playing controller
- playing cricket
- playing cymbals
- playing didgeridoo
- playing drums
- playing flute
- playing guitar
- playing harmonica
- playing harp
- playing ice hockey
- playing keyboard
- playing kickball
- playing monopoly
- playing organ
- playing paintball
- playing piano
- playing poker
- playing recorder
- playing saxophone
- playing squash or racquetball
- playing tennis
- playing trombone
- playing trumpet
- playing ukulele
- playing violin
- playing volleyball
- playing xylophone
- pole vault
- presenting weather forecast
- pull ups
- pumping fist
- pumping gas
- punching bag
- punching person (boxing)
- push up
- pushing car
- pushing cart
- pushing wheelchair
- reading book
- reading newspaper
- recording music
- riding a bike
- riding camel
- riding elephant
- riding mechanical bull
- riding mountain bike
- riding mule
- riding or walking with horse
- riding scooter
- riding unicycle
- ripping paper
- robot dancing
- rock climbing
- rock scissors paper
- roller skating
- running on treadmill
- sailing
- salsa dancing
- sanding floor
- scrambling eggs
- scuba diving
- setting table
- shaking hands
- shaking head
- sharpening knives
- sharpening pencil
- shaving head
- shaving legs
- shearing sheep
- shining shoes
- shooting basketball
- shooting goal (soccer)
- shot put
- shoveling snow
- shredding paper
- shuffling cards
- side kick
- sign language interpreting
- singing
- situp
- skateboarding
- ski jumping
- skiing (not slalom or crosscountry)
- skiing crosscountry
- skiing slalom
- skipping rope
- skydiving
- slacklining
- slapping
- sled dog racing
- smoking
- smoking hookah
- snatch weight lifting
- sneezing
- sniffing
- snorkeling
- snowboarding
- snowkiting
- snowmobiling
- somersaulting
- spinning poi
- spray painting
- spraying
- springboard diving
- squat
- sticking tongue out
- stomping grapes
- stretching arm
- stretching leg
- strumming guitar
- surfing crowd
- surfing water
- sweeping floor
- swimming backstroke
- swimming breast stroke
- swimming butterfly stroke
- swing dancing
- swinging legs
- swinging on something
- sword fighting
- tai chi
- taking a shower
- tango dancing
- tap dancing
- tapping guitar
- tapping pen
- tasting beer
- tasting food
- testifying
- texting
- throwing axe
- throwing ball
- throwing discus
- tickling
- tobogganing
- tossing coin
- tossing salad
- training dog
- trapezing
- trimming or shaving beard
- trimming trees
- triple jump
- tying bow tie
- tying knot (not on a tie)
- tying tie
- unboxing
- unloading truck
- using computer
- using remote controller (not gaming)
- using segway
- vault
- waiting in line
- walking the dog
- washing dishes
- washing feet
- washing hair
- washing hands
- water skiing
- water sliding
- watering plants
- waxing back
- waxing chest
- waxing eyebrows
- waxing legs
- weaving basket
- welding
- whistling
- windsurfing
- wrapping present
- wrestling
- writing
- yawning
- yoga
- zumba
    ''')

o2 = st.checkbox('Info about the data')
if o2:
    st.header("Dataset: The Kinetics Dataset")
    st.image('img/dataset.jpg',use_column_width=True)
    st.markdown('view dataset from <a href="https://arxiv.org/abs/1705.06950" target="blank">link</a>',unsafe_allow_html=True)

o3 = st.radio("select option",("Use AI using webcam for recognition","Use AI in video for recognition"))
if o3 == 'Use AI using webcam for recognition':
    camera = st.number_input("enter the camera number if you have more than one camera, main camera is 0", 0,2,value=0)
    st.warning('keep it 0 if you have only one camera')
    st.warning('if wrong camera number is given app will crash')
    if st.button('start camera'):
        with st.spinner('camera view opens in new window, use "Q" key to close that window'):
            execute(camera)
            st.info("camera closed")
    else:
        st.info('click on button to start camera')

if o3 == 'Use AI in video for recognition':
    video = st.file_uploader('upload a video (less than 25mb)',type=['mp4'])
    if video and st.button('start recongnition'):
        with st.spinner("saving and processing video"):
            with open('videos/video2.mp4','wb') as f:
                f.write(video.read())
            st.info("processed video")
        with st.spinner('video view opens in new window, use "Q" key to close that window'):
            execute('videos/video2.mp4')
            st.info("Video closed")
