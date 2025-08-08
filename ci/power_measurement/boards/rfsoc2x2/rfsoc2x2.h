#define INA226_BUS "/dev/i2c-9"

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
    v0_85,
    v1_2_ps,
    v1_2_pl,
    v1_1,
    v1_8,
    v3_5,
    v3_3,
    syzygy_vio,
    v2_5,
    v5_0,
    NUM_OF_RAILS
};

enum sensors_e {
    NUM_OF_SENSORS
};

void writeData(int fdi2c, unsigned char address, unsigned char reg, int value);
int readData(int fdi2c, unsigned char address, unsigned char reg);
double readBusVoltage(int fdi2c, unsigned char address);
double readCurrent(int fdi2c, unsigned char address);
double readPower(int fdi2c, unsigned char address);

void update_rail(int rail_id);
