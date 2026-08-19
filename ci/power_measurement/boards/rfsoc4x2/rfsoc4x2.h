#define INA220_BUS "/dev/i2c-0" //CMON I2C

//PMBUS Commands
#define CMD_PAGE    0x00
#define REG_CONFIG  0x00
#define REG_SHUNT_V 0x01
#define REG_BUS_V   0x02
#define REG_POWER   0x03
#define REG_CURRENT 0x04
#define REG_CAL     0x05
#define REG_EN      0x06
#define REG_ALERT   0x07
#define REG_ID      0xFE
#define REG_DIE     0xFF

enum rails_e {
    v0_85, //1m
    v3_3, //1m
    v1_8, //1m
    vdac_avcc, //10m
    vadc_avcc, //10m
    vadc_avccaux, // 10m
    vdac_avccaux, //10m
    vdac_avtt, //10m
    syzygy_vio, //10m
    NUM_OF_RAILS
};

enum sensors_e {
    NUM_OF_SENSORS
};

void writeData(int fdi2c, unsigned char address, unsigned char reg, int value);
int readData(int fdi2c, unsigned char address, unsigned char reg);
double readBusVoltage(int fdi2c, int rail_id);
double readCurrent(int fdi2c, int rail_id);
double readPower(int fdi2c, int rail_id);

void update_rail(int rail_id);
