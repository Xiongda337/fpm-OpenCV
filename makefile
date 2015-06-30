CC=g++
IDIR=.
CFLAGS=-I$(IDIR) -std=c++11 -ggdb -fopenmp

LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

fpmMain: fpmMain.cpp
	$(CC) -o fpmMain fpmMain.cpp $(CFLAGS) $(LIBS)

clean:
	rm -rf fpmMain
