#ifndef _AS7341_LIB_
#define _AS7341_LIB_

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* Kernel includes. */
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

/* Hardware includes. */
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "driverlib/adc.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/sysctl.h"
#include "driverlib/pwm.h"
#include "driverlib/pin_map.h"
#include "utils/uartstdio.h"
#include "utils/ustdlib.h"
#include "AS7341/AS7341.h"
#include "driverlib/i2c.h"

// Esse codigo se baseia nos exemplos presente em 

// ./Tiva/EK-T4MC/examples/peripherals/i2c/slave_receive_int.c &
// ~/worksapece-v10/hello/external_devices/AS7341_photo.c &
// ./Tiva/EK-TM4C/third_party/FreeRTOS/Demo/CORTEX_LM3S102_Rowley/Demo3/main.c
SemaphoreHandle_t I2C1_Semphr;
void I2C1_IntHandler(void){
  I2CMasterIntClear( I2C1_BASE ); 
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  xSemaphoreGiveFromISR(I2C1_Semphr, &xHigherPriorityTaskWoken);
}

bool WaitACK(){
  if(xSemaphoreTake(I2C1_Semphr, pdMS_TO_TICKS(1500))==pdFALSE)
    return false;
  else
    return true;
}

bool AS7341_write(uint8_t regAdd, uint8_t data){
  
  //Write Operation
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, false);
  I2CMasterDataPut(I2C1_BASE, regAdd);
  //START
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_START);

  //Wait ACK
  if(!WaitACK()) return false;

  I2CMasterDataPut(I2C1_BASE, data);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);

  //Wait ACK
  if(!WaitACK()) return false;
  return true;
}
}
bool AS7341_writeMultiples(uint8_t startReg, uint8_t *data, uint32_t length ){
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, false);
  I2CMasterDataPut(I2C1_BASE, startReg);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_START);
  if(!WaitACK()) return false;

  for(uint32_t index=0;index<length; index++){
    I2CMasterDataPut(I2C1_BASE, data[index]);

    if(index==(length-1))
      I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);
    else
      I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_CONT);
    
    if(!WaitACK()) return false;
  }
  
  return true;

}

bool AS7341_read(uint8_t regAdd, uint8_t *data){
  
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, false);
  I2CMasterDataPut(I2C1_BASE, regAdd);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_START);

  if(!WaitACK()) return false;
  
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, true);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_SINGLE_RECEIVE);

  if(!WaitACK()) return false;

  *data = I2CMasterDataGet(I2C1_BASE);
  return true;
}
bool AS7341_readMultiples( uint8_t startReg, uint8_t *data, uint32_t length){
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, false);
  I2CMasterDataPut(I2C1_BASE, regAdd);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_START);

  if(!WaitACK()) return false;
  
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, true);
  for(uint32_t index=0;index<length; index++){
    I2CMasterDataPut(I2C1_BASE, data[index]);

    if(index==(length-1))
      I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_RECEIVE_FINISH);
    else
        I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_RECEIVE_CONT);
    
    if(!WaitACK()) return false;
    data[index] = I2CMasterDataGet(I2C1_BASE);
  }

  return true;
}

bool AS7341_Init(){
  I2C1_Semphr = xSemaphoreCreateMutex();
  SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C1);
  
  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);
  
  GPIOPinConfigure(GPIO_PA6_I2C1SCL);
  GPIOPinConfigure(GPIO_PA7_I2C1SDA);
  GPIOPinTypeI2CSCL(GPIO_PORTA_BASE, GPIO_PIN_6);
  GPIOPinTypeI2C(GPIO_PORTA_BASE, GPIO_PIN_7);

  I2CMasterInitExpClk(I2C1_BASE, SysCtlClockGet(), false);
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, false);
  I2CMasterIntEnable( I2C1_BASE );
  IntEnable(INT_I2C1);
  return true;

}

bool AS7341_Enable(){
  return true;

}
bool AS7341_DevivceConfig(){
  return true;

}
bool AS7341_ADC_TimingConfig(){
  return true;

}
bool AS7341_ADC_Config(){
  return true;

}
bool AS7341_InterruptionConfig(){
  return true;

}
bool AS7341_DeviceStatus(){
  return true;

}
bool AS7341_SpecData(){
  return true;

}
bool AS7341_SpecStatus(){
  return true;

}
bool AS7341_OtherConfig(){
  return true;

}
bool AS7341_BufferData(){
  return true;

}
bool AS7341_BufferConfig(){
  return true;

}

#endif
