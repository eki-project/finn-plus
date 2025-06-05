/**
 * Modified version of code by Xilinx forum user "pmpakos": https://forums.xilinx.com/t5/Xilinx-Evaluation-Boards/Power-monitoring-through-Linux-application-ZCU102/td-p/810128
 * Based on https://github.com/witjaksana/zcu_power_monitor which itself is based on https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841995/Zynq-7000+AP+SoC+Low+Power+Techniques+part+4+-+Measuring+ZC702+Power+with+a+Linux+Application+Tech+Tip
*/

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>

#include "../i2c-dev.h"
#include "../board_interface.h"
#include "rfsoc2x2.h"


struct rail *rails[NUM_OF_RAILS]; //filled in initalize()
//struct sensor *temp_sensors[NUM_OF_SENSORS]; //filled in initialize

struct ina226 {
    char* bus;
    unsigned char address;
    int calibration_value;
    struct rail s_rail;
};

struct ina226 sensors[] = {
    {
        bus     :   INA226_BUS,
        address :   0x40,
        calibration_value : 0x1400,
        s_rail : {rail_name : "0V85", id : v0_85, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA226_BUS,
        address :   0x41,
        calibration_value : 0x1400,
        s_rail : {rail_name : "1V2_PS", id : v1_2_ps, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA226_BUS,
        address :   0x42,
        calibration_value : 0x1400,
        s_rail : {rail_name : "1V2_PL", id : v1_2_pl, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA226_BUS,
        address :   0x43,
        calibration_value : 0x1400,
        s_rail : {rail_name : "1V1_DC", id : v1_1, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA226_BUS,
        address :   0x45,
        calibration_value : 0x1400,
        s_rail : {rail_name : "1V8", id : v1_8, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA226_BUS,
        address :   0x47,
        calibration_value : 0x1400,
        s_rail : {rail_name : "3V5_DC", id : v3_5, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA226_BUS,
        address :   0x48,
        calibration_value : 0x1400,
        s_rail : {rail_name : "3V3", id : v3_3, voltage : 0.0, current : 0.0, power : 0.0, update_rail},
    },
    {
        bus     :   INA226_BUS,
        address :   0x49,
        calibration_value : 0x1400,
        s_rail : {rail_name : "SYZYGY_VIO", id : syzygy_vio, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA226_BUS,
        address :   0x4C,
        calibration_value : 0x1400,
        s_rail : {rail_name : "2V5_DC", id : v2_5, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA226_BUS,
        address :   0x4D,
        calibration_value : 0x1400,
        s_rail : {rail_name : "5V0_DC", id : v5_0, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    }
};

int get_num_rails() {
    return NUM_OF_RAILS;
}

struct rail** get_rails() {
    return rails;
}

int get_num_sensors() {
    return NUM_OF_SENSORS;
}

struct sensor **get_sensors() {
    return NULL;
}

void update_rail(int rail_id) {
    int fdi2c;
    fdi2c = open(sensors[rail_id].bus, O_RDWR);
    if(fdi2c < 0) {
        return;
    }

    sensors[rail_id].s_rail.voltage = readBusVoltage(fdi2c, sensors[rail_id].address);
    sensors[rail_id].s_rail.current = readCurrent(fdi2c, sensors[rail_id].address);
    sensors[rail_id].s_rail.power = sensors[rail_id].s_rail.voltage*sensors[rail_id].s_rail.current;

    close(fdi2c);
}

void writeData(int fdi2c, unsigned char address, unsigned char reg, int value){
    int status;

    if (ioctl(fdi2c, I2C_SLAVE_FORCE, address) < 0){
        printf("ERROR: Unable to set I2C slave address 0x%02X\n", address);
        exit(1);
    }

    status = i2c_smbus_write_byte_data(fdi2c, CMD_PAGE, address);
    if (status < 0) {
        printf("ERROR: Unable to write page address to I2C slave at 0x%02X: %d\n", address, status);
        exit(1);
    }

    value = (value >> 8) | ((value & 0xff) << 8); /** turn the byte around */

    status = i2c_smbus_write_word_data(fdi2c, reg, value);
    if (status < 0) {
        printf("ERROR: Unable to write value to I2C reg at 0x%02X: %d\n", reg, status);
        exit(1);
    }
}

int readData(int fdi2c, unsigned char address, unsigned char reg){
    int status;
    int value;

    if (ioctl(fdi2c, I2C_SLAVE_FORCE, address) < 0){
        printf("ERROR: Unable to set I2C slave address 0x%02X\n", address);
        exit(1);
    }

    status = i2c_smbus_write_byte_data(fdi2c, CMD_PAGE, address);
    if (status < 0) {
        printf("ERROR: Unable to write page address to I2C slave at 0x%02X: %d\n", address, status);
        exit(1);
    }

    value = i2c_smbus_read_word_data(fdi2c, reg);
    value = (value >> 8) | ((value & 0xff) << 8); /** turn the byte around */
    return value;
}

double readBusVoltage(int fdi2c, unsigned char address){
    int raw_value;
    double voltage;

    raw_value = readData(fdi2c, address, REG_BUS_V);

    voltage = (float)raw_value * 0.00125;
    return voltage;
}

double readCurrent(int fdi2c, unsigned char address){
    int raw_value;
    double current;

    raw_value = readData(fdi2c, address, REG_CURRENT);
    // in case it's negative
    if ((raw_value & 0x8000) != 0){
        raw_value |= 0xffff0000;
    }

    current = (float)raw_value;
    return current;
}

double readPower(int fdi2c, unsigned char address){
    int raw_value;
    double power;

    raw_value = readData(fdi2c, address, REG_POWER);

    power = (float)raw_value * 0.025;
    return power;
}

int initialize() {
    //Fill rails array
    for(int i = 0; i < NUM_OF_RAILS; i++) {
        rails[i] = &(sensors[i].s_rail);
    }

    //Calibrate ina226
    printf("Calibrating INA226 sensors...\n");
    for(int i = 0; i < NUM_OF_RAILS; i++) {
        int fdi2c;
        fdi2c = open(sensors[i].bus, O_RDWR);
        if(fdi2c < 0) {
            printf("ERROR: Opening %s failed\n", sensors[i].bus);
            return -1;
        }

        //printf("bus: %s name: %s addr: 0x%02X cal_val: 0x%04X\n", sensors[i].bus, sensors[i].s_rail.rail_name, sensors[i].address, sensors[i].calibration_value);
        writeData(fdi2c, sensors[i].address, REG_CAL, sensors[i].calibration_value);
        close(fdi2c);
    }
    printf("Done calibrating INA226 sensors.\n");
    return 0;
}

int main(int argc, char* argv[]) {
    return initialize();
}
