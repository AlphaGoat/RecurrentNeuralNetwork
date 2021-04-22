CC=g++
CFLAGS=-g -Wall -I. -I./utils
OBJECTS=main.o
DEPS=utils.h

VOWEL_FLAGS=-DNUM_INPUTS=12 -DNUM_OUTPUTS=9 -DNUM_RECURRENT_UNITS=16
SIGN_FLAGS=-DNUM_INPUTS=10 -DNUM_OUTPUTS=5 -DNUM_RECURRENT_UNITS=12
IRIS_FLAGS=-DNUM_INPUTS=4 -DNUM_OUTPUTS=3 -DNUM_RECURRENT_UNITS=3

all: trainVowels trainSigns trainIris
#all: trainIris
#all: trainVowels
#all: trainSigns

trainSigns: trainSigns.o 
	$(CC) $(CFLAGS) $(SIGN_FLAGS) -o trainSigns trainSigns.cpp utils/utils.cpp utils/ReadCSV.cpp utils/median_filter.cpp utils/sign_utils.cpp model/RNNBase.cpp model/MitchellRNNv2.cpp model/RecurrentNeuralNetworkv2.cpp 

trainVowels: trainVowels.o
	$(CC) $(CFLAGS) $(VOWEL_FLAGS) -o trainVowels trainVowels.cpp utils/utils.cpp utils/ReadCSV.cpp utils/vowel_utils.cpp model/MitchellRNNv2.cpp model/RecurrentNeuralNetworkv2.cpp model/RNNBase.cpp
#
trainIris: trainIris.o 
	$(CC) $(CFLAGS) $(IRIS_FLAGS) -o trainIris trainIris.cpp utils/utils.cpp utils/ReadCSV.cpp model/MitchellRNNv2.cpp model/RecurrentNeuralNetworkv2.cpp model/RNNBase.cpp

clean:
	rm -f trainIris trainIris.o
	rm -f trainSigns trainSigns.o 
	rm -f trainVowels trainVowels.o
