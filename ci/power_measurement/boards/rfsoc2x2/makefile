rfsoc2x2_test: rfsoc2x2.c
	gcc -Wall rfsoc2x2.c -o rfsoc2x2 -lm

perm:
	sudo chmod a+rw /dev/i2c-9

shared: rfsoc2x2.c
	gcc -shared -o rfsoc2x2.so -fPIC rfsoc2x2.c -lm

clean:
	rm _sourceout.*
	rm rfsoc2x2.o
