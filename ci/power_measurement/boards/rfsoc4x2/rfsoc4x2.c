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
#include <stdbool.h>

#include "../i2c-dev.h"
#include "../board_interface.h"
#include "rfsoc4x2.h"

struct rail *rails[NUM_OF_RAILS]; //filled in initalize()
//struct sensor *temp_sensors[NUM_OF_SENSORS]; //filled in initialize


struct ina220 {
    char* bus;
    unsigned char address;
    double current_lsb; // in mA
    double resistor_shunt; //in mOhm
    struct rail s_rail;
};

struct ina220 sensors[] = {
    {
        bus     :   INA220_BUS,
        address :   0x40,
        current_lsb : 2,
        resistor_shunt : 1,
        s_rail : {rail_name : "0V85", id : v0_85, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA220_BUS,
        address :   0x41,
        current_lsb : 2,
        resistor_shunt : 1,
        s_rail : {rail_name : "3V3", id : v3_3, voltage : 0.0, current : 0.0, power : 0.0, update_rail}

    },
    {
        bus     :   INA220_BUS,
        address :   0x42,
        current_lsb : 2,
        resistor_shunt : 1,
        s_rail : {rail_name : "1V8", id : v1_8, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA220_BUS,
        address :   0x43,
        current_lsb : 2,
        resistor_shunt : 10,
        s_rail : {rail_name : "VDAC_AVCC", id : vdac_avcc, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA220_BUS,
        address :   0x44,
        current_lsb : 2,
        resistor_shunt : 10,
        s_rail : {rail_name : "VADC_AVCC", id : vadc_avcc, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA220_BUS,
        address :   0x45,
        current_lsb : 2,
        resistor_shunt : 10,
        s_rail : {rail_name : "VADC_AVCCAUX", id : vadc_avccaux, voltage : 0.0, current : 0.0, power : 0.0, update_rail},
    },
    {
        bus     :   INA220_BUS,
        address :   0x46,
        current_lsb : 2,
        resistor_shunt : 10,
        s_rail : {rail_name : "VDAC_AVCCAUX", id : vdac_avccaux, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA220_BUS,
        address :   0x47,
        current_lsb : 2,
        resistor_shunt : 10,
        s_rail : {rail_name : "VDAC_AVTT", id : vdac_avtt, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
    },
    {
        bus     :   INA220_BUS,
        address :   0x48,
        current_lsb : 2,
        resistor_shunt : 10,
        s_rail : {rail_name : "SYZYGY_VIO", id : syzygy_vio, voltage : 0.0, current : 0.0, power : 0.0, update_rail}
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

    sensors[rail_id].s_rail.voltage = readBusVoltage(fdi2c, rail_id);
    sensors[rail_id].s_rail.current = readCurrent(fdi2c, rail_id);
    sensors[rail_id].s_rail.power = sensors[rail_id].s_rail.voltage * sensors[rail_id].s_rail.current;

    close(fdi2c);
}

void writeData(int fdi2c, unsigned char address, unsigned char reg, int value){
    int status;

    if (ioctl(fdi2c, I2C_SLAVE_FORCE, address) < 0){
        printf("[I2C Driver] ERROR: Unable to set I2C slave address 0x%02X\n", address);
        exit(1);
    }

    status = i2c_smbus_write_byte_data(fdi2c, CMD_PAGE, address);
    if (status < 0) {
        printf("[I2C Driver] ERROR: Unable to write page address to I2C slave at 0x%02X: %d\n", address, status);
        exit(1);
    }

    value = (value >> 8) | ((value & 0xff) << 8); /** turn the byte around */

    status = i2c_smbus_write_word_data(fdi2c, reg, value);
    if (status < 0) {
        printf("[I2C Driver] ERROR: Unable to write value to I2C reg at 0x%02X: %d\n", reg, status);
        exit(1);
    }
}

int readData(int fdi2c, unsigned char address, unsigned char reg){
    int status;
    int value;

    if (ioctl(fdi2c, I2C_SLAVE_FORCE, address) < 0){
        printf("[I2C Driver] ERROR: Unable to set I2C slave address 0x%02X\n", address);
        exit(1);
    }

    status = i2c_smbus_write_byte_data(fdi2c, CMD_PAGE, address);
    if (status < 0) {
        printf("[I2C Driver] ERROR: Unable to write page address to I2C slave at 0x%02X: %d\n", address, status);
        exit(1);
    }

    value = i2c_smbus_read_word_data(fdi2c, reg);
    value = (value >> 8) | ((value & 0xff) << 8); /** turn the byte around */
    return value;
}

double readBusVoltage(int fdi2c, int rail_id){
    int raw_value;
    double voltage;

    bool cnvr = false;
    int bus_volt = 0;

    for (int i = 0; i < 50; i++)
    {
        raw_value = readData(fdi2c, sensors[rail_id].address, REG_BUS_V);
        bool ovf = raw_value & 0x01; // Math Overflow Flag
        cnvr = (raw_value >> 1) & 0x01 ;
        bus_volt = (raw_value >> 3);

        if(ovf == true)
        {
            printf("[I2C Driver] ERROR: Bus voltage overflow");
            exit(1);
        }

        if(cnvr == true)
        {
            break;
        }
    }

    if(cnvr == false)
    {
        printf("[I2C Driver] ERROR: Unable to read bus voltage register");
        exit(1);
    }

    voltage = (float)bus_volt * 0.004; // LSB is 4mV as per Datasheet
    return voltage;
}

double readCurrent(int fdi2c, int rail_id){
    int raw_value;
    double current;

    raw_value = readData(fdi2c, sensors[rail_id].address, REG_CURRENT);
    // in case it's negative
    if ((raw_value & 0x8000) != 0){
        raw_value |= 0xffff0000;
    }

    current = (float)raw_value * 0.001 * sensors[rail_id].current_lsb;
    return current;
}

double readPower(int fdi2c, int rail_id){
    int raw_value;
    double power;

    raw_value = readData(fdi2c, sensors[rail_id].address, REG_POWER);

    power = (float)raw_value * (20 * 0.001 * sensors[rail_id].current_lsb); // Power_LSB = 20 * Current_LSB (datasheet)
    return power;
}

unsigned int calculateCalibration(double current_lsb, double resistor_shunt){
    // CAL = (0.04096 / (Current_LSB * R_shunt)) (From datasheet)
    double full_cal = 40960 / (current_lsb * resistor_shunt);
    unsigned int trunc_cal = (unsigned int)full_cal;
    if (trunc_cal > 0x7FFF) // 15 Bit for calibration value
    {
        printf("[I2C Driver] Warning: INA220 Calibration Overflow!!\n");
        trunc_cal = 0x7FFF;
    }

    return (trunc_cal << 1); //Shift by 1 because bit 0 is ununsed
}

int initialize() {
    // Disable stdout buffering
    setvbuf(stdout, NULL, _IONBF, 0);

    //Fill rails array
    for(int i = 0; i < NUM_OF_RAILS; i++) {
        rails[i] = &(sensors[i].s_rail);
    }

    //Calibrate INA220
    for(int i = 0; i < NUM_OF_RAILS; i++) {
        int fdi2c;
        fdi2c = open(sensors[i].bus, O_RDWR);
        if(fdi2c < 0) {
            printf("[I2C Driver] ERROR: Opening %s failed\n", sensors[i].bus);
            return -1;
        }

        unsigned int calibration_value = calculateCalibration(sensors[i].current_lsb, sensors[i].resistor_shunt);
        writeData(fdi2c, sensors[i].address, REG_CAL, calibration_value);
        close(fdi2c);
    }
    printf("[I2C Driver] Calibrated %d INA220 sensors.\n", NUM_OF_RAILS);
    return 0;
}

int main(int argc, char* argv[]) {
    return initialize();
}
