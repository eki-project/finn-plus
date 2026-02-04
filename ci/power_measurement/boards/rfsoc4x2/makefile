rfsoc4x2_test: rfsoc4x2.c
	gcc -Wall rfsoc4x2.c -o rfsoc4x2 -lm

perm:
	sudo chmod a+rw /dev/i2c-9

shared: rfsoc4x2.c
	gcc -shared -o rfsoc4x2.so -fPIC rfsoc4x2.c -lm

clean:
	rm _sourceout.*
	rm rfsoc4x2.o
