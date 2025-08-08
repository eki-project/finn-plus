#include <stdio.h>

struct sensor {
    char* name;
    int id;
    char* unit;
    double value;
    void (*update_value)(int id);
};

struct rail {
    char* rail_name;
    int id;
    double voltage;
    double current;
    double power;
    void (*update_values)(int id);
};

int initialize();

struct rail** get_rails();
int get_num_rails(); //returns number of rails

struct sensor** get_sensors();
int get_num_sensors();
