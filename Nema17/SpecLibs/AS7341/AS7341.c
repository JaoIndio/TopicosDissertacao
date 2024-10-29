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
uint8_t BankAcessControlValue = 0;
SemaphoreHandle_t I2C1_Semphr;
SemaphoreHandle_t AS7341_Semphr;

void CheckArray(uint8_t *photo){
  uint8_t index =0;
  UARTprintf("\n");
  while(1){
    UARTprintf("- %d - ", photo[index]);
    index++;
    if(index==18) 
      break;
  }
  UARTprintf("\n");
}

void PortDIntHanlder(){
  GPIOIntClear(GPIO_PORTD_BASE, GPIO_PIN_0);
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  xSemaphoreGiveFromISR(AS7341_Semphr, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void I2C1_IntHandler(){
  I2CMasterIntClear( I2C1_BASE ); 
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  xSemaphoreGiveFromISR(I2C1_Semphr, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

bool WaitACK(){
  if(xSemaphoreTake(I2C1_Semphr, pdMS_TO_TICKS(25))==pdFALSE)
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
  //while(I2CMasterBusy(I2C1_BASE));

  I2CMasterDataPut(I2C1_BASE, data);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_FINISH);

  //Wait ACK
  if(!WaitACK()) return false;
  //while(I2CMasterBusy(I2C1_BASE));
  return true;
}
bool AS7341_writeMultiples(uint8_t startReg, uint8_t *data, uint32_t length ){
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, false);
  I2CMasterDataPut(I2C1_BASE, startReg);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_START);
  if(!WaitACK()) return false;
  uint32_t index =0;

  for(index=0;index<length; index++){
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
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_SINGLE_SEND);

  if(!WaitACK()) return false;
  
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, true);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_SINGLE_RECEIVE);

  if(!WaitACK()) return false;

  *data = I2CMasterDataGet(I2C1_BASE);
  return true;
}
bool AS7341_readMultiples( uint8_t startReg, uint8_t *data, uint32_t length){

  uint8_t data_copy[12];
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, false);
  I2CMasterDataPut(I2C1_BASE, startReg);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_SEND_START);
  if(!WaitACK()) 
    return false;
  uint32_t index;
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, true);

  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_RECEIVE_START);
  if(!WaitACK())
    return false;

  data[0] = I2CMasterDataGet(I2C1_BASE);

  for(index=1;index<length; index++){
    //I2CMasterDataPut(I2C1_BASE, data[index]);

    if(index==(length-1)){
      I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_RECEIVE_FINISH);
      //break;
    }else
        I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_BURST_RECEIVE_CONT);
    
    if(!WaitACK()) 
      return false;
    data[index] = I2CMasterDataGet(I2C1_BASE);
  }
  
  for(index=0; index<length; index++)
    data_copy[index] = data[index];
  return true;
}

bool AS7341_i2cInit(){
  I2C1_Semphr = xSemaphoreCreateBinary();
  SysCtlPeripheralEnable(SYSCTL_PERIPH_I2C1);
  while(!SysCtlPeripheralReady(SYSCTL_PERIPH_I2C1))
  {
  }
  
  //SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);
  GPIOUnlockPin(GPIO_PORTA_BASE, GPIO_PIN_6|GPIO_PIN_7);

  GPIOPinConfigure(GPIO_PA6_I2C1SCL);
  GPIOPinConfigure(GPIO_PA7_I2C1SDA);
  GPIOPinTypeI2CSCL(GPIO_PORTA_BASE, GPIO_PIN_6);
  //GPIOPadConfigSet(GPIO_PORTA_BASE, GPIO_PIN_6 | GPIO_PIN_7, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_OD);
  //GPIOPinTypeI2C(GPIO_PORTA_BASE, GPIO_PIN_7|GPIO_PIN_6);
  GPIOPinTypeI2C(GPIO_PORTA_BASE, GPIO_PIN_7);

  I2CMasterInitExpClk(I2C1_BASE, SysCtlClockGet(), false);
  I2CMasterIntEnableEx( I2C1_BASE, I2C_MASTER_INT_DATA );
  IntEnable(INT_I2C1);

  // Enables PD0 to handle AS7341 Interruptions
  AS7341_Semphr = xSemaphoreCreateBinary();
  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
  GPIOUnlockPin(GPIO_PORTD_BASE, GPIO_PIN_0 | GPIO_PIN_1);
  GPIOPinTypeGPIOInput(GPIO_PORTD_BASE, GPIO_PIN_0);
  GPIOIntTypeSet(GPIO_PORTD_BASE, GPIO_PIN_0, GPIO_FALLING_EDGE);
  GPIOIntClear(GPIO_PORTD_BASE, GPIO_PIN_0);
  GPIOIntEnable(GPIO_PORTD_BASE, GPIO_INT_PIN_0);
  IntEnable(INT_GPIOD);
  IntMasterEnable();

  GPIOPinTypeGPIOOutput(GPIO_PORTD_BASE, GPIO_PIN_2);
  GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_2, GPIO_PIN_2);

  return true;

}


bool AS7341_Enable(){
  as7341_enable_t enable_reg;
  if(!AS7341_SetAcessAndRead(AS7341_REG_ENABLE, &enable_reg.value)) return false;
  enable_reg.PON    = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ENABLE, enable_reg.value)) return false;
  vTaskDelay(pdMS_TO_TICKS(1));

  as7341_status6_t status6_reg;
  while(1){
    AS7341_SetAcessAndRead(AS7341_REG_STATUS6, &status6_reg.value);
    if(!status6_reg.INT_BUSY) break;
    vTaskDelay(pdMS_TO_TICKS(2));
  }
  AS7341_SetAcessAndWrite(AS7341_REG_STATUS6, status6_reg.value);
  return true;
}
bool AS7341_DevivceConfig(){
  
  as7341_config_t config_reg;
  config_reg.LED_SEL  = 1;
  config_reg.INT_SEL  = 0;
  config_reg.INT_MODE = 0;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CONFIG, config_reg.value)) return false;
  
  as7341_led_t led_reg;
  led_reg.LED_ACT   = 0;
  led_reg.LED_DRIVE = 0;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_LED, led_reg.value)) return false;
/*
  as7341_gpio_t gpio_reg;
  gpio_reg.PD_INT   = 0;
  gpio_reg.PD_GPIO  = 0;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_GPIO1, gpio_reg.value)) return false;

  as7341_gpio2_t gpio2_reg;
  gpio2_reg.GPIO_INV   = 0;
  gpio2_reg.GPIO_IN_EN = 1;
  gpio2_reg.GPIO_OUT   = 0;
  gpio2_reg.GPIO_IN    = 0;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_GPIO2, gpio2_reg.value)) return false;
  
  
  as7341_intenab_t intenab_reg;
  intenab_reg.ASIEN   = 0;
  intenab_reg.SP_IEN  = 0;
  intenab_reg.F_IEN   = 0;
  intenab_reg.SIEN    = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_INTENAB, intenab_reg.value)) return false;
  
  as7341_control_t control_reg;
  control_reg.SP_MAN_AZ       = 0;
  control_reg.FIFO_CLR        = 0; //Talvez valha pena ter uma funcao so pra esse cmd
  control_reg.CLEAR_SAI_ACT   = 0;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CONTROL, control_reg.value)) return false;
*/  
  return true;
}
bool AS7341_ADC_TimingConfig(){
  //𝑡𝑖𝑛𝑡 = (𝐴𝑇𝐼𝑀𝐸 + 1) × (𝐴𝑆𝑇𝐸𝑃 + 1) × 2.78μ𝑠
  // 𝐴𝐷𝐶𝑓𝑢𝑙𝑙𝑠𝑐𝑎𝑙𝑒 = (𝐴𝑇𝐼𝑀𝐸 + 1) × (𝐴𝑆𝑇𝐸𝑃 + 1)
  as7341_atime_t atime_reg;
  atime_reg.ATIME   = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ATIME, atime_reg.value)) return false;
  
  as7341_astep_t astep_reg;
  astep_reg.ASTEP_L   = 1;
  astep_reg.ASTEP_H   = 0;                     
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ASTEP_L, (uint8_t)astep_reg.value))      return false;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ASTEP_H, (uint8_t)(astep_reg.value>>8))) return false;
  
  as7341_wtime_t wtime_reg;
  wtime_reg.WTIME   = 0; // 2,78ms
  if(!AS7341_SetAcessAndWrite(AS7341_REG_WTIME, wtime_reg.value)) return false;

  return true;

}
bool AS7341_ADC_Config(){
  as7341_cfg1_t cfg1_reg;
  cfg1_reg.AGAIN   = 7; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG1, cfg1_reg.value)) return false;
  
  as7341_cfg10_t cfg10_reg;
  cfg10_reg.AGC_H   = 3; //
  cfg10_reg.AGC_L   = 0; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG10, cfg10_reg.value)) return false;

  as7341_az_config_t az_config_reg;
  az_config_reg.AZ_NTH_ITERATION   = 25; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_AZ_CONFIG, az_config_reg.value)) return false;

  as7341_agc_gain_max_t agc_gain_max_reg;
  agc_gain_max_reg.AGC_AGAIN_MAX = 10;
  agc_gain_max_reg.AGC_FD_GAIN_MAX = 9;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_AGC_GAIN_MAX, agc_gain_max_reg.value)) return false;

  as7341_cfg8_t cfg8_reg;
  cfg8_reg.FIFO_TH   = 3; //
  cfg8_reg.FD_AGC    = 0; //
  cfg8_reg.SP_AGC    = 0; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG8, cfg8_reg.value)) return false;
  
  return true;

}

bool AS7341_SetTimeADC(uint8_t value){
  as7341_atime_t atime_reg;
  atime_reg.ATIME = value;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ATIME, atime_reg.value)) return false;
  return true;
}

uint16_t AS7341_GetStepADC(){
  as7341_astep_t astep_reg;
  uint8_t generic;
  AS7341_SetAcessAndRead(AS7341_REG_ASTEP_H, &generic);
  astep_reg.ASTEP_H = generic;
  AS7341_SetAcessAndRead(AS7341_REG_ASTEP_L,&generic);
  astep_reg.ASTEP_L = generic;

  return astep_reg.value;
}

uint16_t AS7341_GetTimeADC(){
  as7341_atime_t atime_reg;
  AS7341_SetAcessAndRead(AS7341_REG_ATIME, &atime_reg.value);
  return atime_reg.value;
}

float AS7341_GetIntegrationTimeADC(){
  uint16_t time = AS7341_GetTimeADC()+1;
  uint16_t step = AS7341_GetStepADC()+1;
  return (float)(step*time*2.78/1000000);
}

bool AS7341_SetStepADC(uint16_t value){
  as7341_astep_t astep_reg;
  astep_reg.ASTEP_L =  value & 0xFF;
  astep_reg.ASTEP_H = (value & 0xFF00)>>8;

  if(!AS7341_SetAcessAndWrite(AS7341_REG_ASTEP_L, astep_reg.ASTEP_L)) return false;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ASTEP_H, astep_reg.ASTEP_H)) return false;
  return true;
}

bool AS7341_SetWtimeADC(uint8_t value){
  as7341_wtime_t wtime_reg;
  wtime_reg.WTIME = value;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_WTIME, wtime_reg.value)) return false;
  return true;
}

bool AS7341_SetGainADC(uint8_t value){
  as7341_cfg1_t cfg1_reg;
  // 1111 1
  cfg1_reg.AGAIN = value & 0x1F;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG1, cfg1_reg.value)) return false;
  return true;
}

bool AS7341_InterruptionConfig(){
  return true;

}
bool AS7341_DeviceStatus(uint8_t regAdd, uint8_t* data){
  if(regAdd==AS7341_REG_STAT    || regAdd==AS7341_REG_STATUS  ||\
     regAdd==AS7341_REG_STATUS2 || regAdd==AS7341_REG_STATUS3 ||\
     regAdd==AS7341_REG_STATUS5 || regAdd==AS7341_REG_STATUS6 ||\
     regAdd==AS7341_REG_FD_STATUS)
    if(!AS7341_SetAcessAndRead(regAdd, data)) return false;
   
  return true;

}
bool AS7341_SpecData(uint8_t regAdd, uint8_t* data){
  if(regAdd==AS7341_REG_ASTATUS1   || regAdd==AS7341_REG_ASTATUS2   ||\
     regAdd==AS7341_REG_CH0_DATA_L || regAdd==AS7341_REG_CH0_DATA_H ||\
     regAdd==AS7341_REG_CH1_DATA_L || regAdd==AS7341_REG_CH1_DATA_H ||\
     regAdd==AS7341_REG_CH2_DATA_L || regAdd==AS7341_REG_CH2_DATA_H ||\
     regAdd==AS7341_REG_CH3_DATA_L || regAdd==AS7341_REG_CH3_DATA_H ||\
     regAdd==AS7341_REG_CH4_DATA_L || regAdd==AS7341_REG_CH4_DATA_H ||\
     regAdd==AS7341_REG_CH5_DATA_L || regAdd==AS7341_REG_CH5_DATA_H)
    if(!AS7341_SetAcessAndRead(regAdd, data)) return false;
  
  return true;
}
bool AS7341_SpecStatus(){
  return true;

}
bool AS7341_OtherConfig(){
  as7341_cfg0_t cfg0_reg;
  cfg0_reg.LOW_POWER = 0; //
  cfg0_reg.REG_BANK  = 0; // fazer funcao especifica para esse reg
  cfg0_reg.WLONG     = 0; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG0, cfg0_reg.value)) return false;
  
  as7341_cfg3_t cfg3_reg;
  cfg3_reg.SAI = 0; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG3, cfg3_reg.value)) return false;
  
  as7341_cfg6_t cfg6_reg;
  cfg6_reg.SMUX_CMD = 2; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG6, cfg6_reg.value)) return false;

  as7341_cfg9_t cfg9_reg;
  cfg9_reg.SIEN_FD   = 0; //
  cfg9_reg.SIEN_SMUX = 1; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG9, cfg9_reg.value)) return false;

  as7341_pers_t pers_reg;
  pers_reg.APERS   = 4; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_PERS, pers_reg.value)) return false;
  
  return true;
}
bool AS7341_BufferData(uint8_t regAdd, uint8_t* data){
  if(regAdd==AS7341_REG_FIFO_LVL || regAdd==AS7341_REG_FDATA_L  ||\
     regAdd==AS7341_REG_FDATA_H )
    if(!AS7341_SetAcessAndRead(regAdd, data)) return false;
  
  return true;

}
bool AS7341_BufferConfig(){
  as7341_fifo_map_t fifo_map_reg;
  fifo_map_reg.FIFO_WRITE_CH0_DATA   = 0; //
  fifo_map_reg.FIFO_WRITE_CH1_DATA   = 0; //
  fifo_map_reg.FIFO_WRITE_CH2_DATA   = 0; //
  fifo_map_reg.FIFO_WRITE_CH3_DATA   = 0; //
  fifo_map_reg.FIFO_WRITE_CH4_DATA   = 0; //
  fifo_map_reg.FIFO_WRITE_CH5_DATA   = 0; //
  fifo_map_reg.FIFO_WRITE_ASTATUS    = 0; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_FIFO_MAP, fifo_map_reg.value)) return false;
  
  as7341_fifo_cfg0_t fifo_cfg0_reg;
  fifo_cfg0_reg.FIFO_WRITE_FD   = 0; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_FIFO_CFG0, fifo_cfg0_reg.value)) return false;

  return true;
}

bool AS7341_EnableSpecMen(){
  as7341_enable_t enable_reg;
  if(!AS7341_SetAcessAndRead(AS7341_REG_ENABLE, &enable_reg.value)) return false;
  enable_reg.SP_EN = 1;
  if(!AS7341_write(AS7341_REG_ENABLE, enable_reg.value)) return false;

  return true;

}
bool AS7341_DisableSpecMen(){
  as7341_enable_t enable_reg;
  if(!AS7341_SetAcessAndRead(AS7341_REG_ENABLE, &enable_reg.value)) return false;
  enable_reg.SP_EN = 0;
  if(!AS7341_write(AS7341_REG_ENABLE, enable_reg.value)) return false;

  return true;
}

bool AS7341_PowerOn(){
  as7341_enable_t enable_reg;

  enable_reg.FDEN   = 0;
  enable_reg.SMUXEN = 0;
  enable_reg.WEN    = 0;
  enable_reg.SP_EN  = 0;
  enable_reg.PON    = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ENABLE, enable_reg.value)) return false;
  vTaskDelay(pdMS_TO_TICKS(1));

  as7341_status6_t status6_reg;
  while(1){
    AS7341_SetAcessAndRead(AS7341_REG_STATUS6, &status6_reg.value);
    if(!status6_reg.INT_BUSY) break;
    vTaskDelay(pdMS_TO_TICKS(2));
  }
  AS7341_SetAcessAndWrite(AS7341_REG_STATUS6, status6_reg.value);
  return true;
}
bool AS7341_PowerOff(){
  as7341_enable_t enable_reg;
  enable_reg.FDEN   = 0;
  enable_reg.SMUXEN = 0;
  enable_reg.WEN    = 0;
  enable_reg.SP_EN  = 0;
  enable_reg.PON    = 1;
  if(!AS7341_SetAcessAndRead(AS7341_REG_ENABLE, &enable_reg.value)) return false;
  enable_reg.PON = 0;
  if(!AS7341_write(AS7341_REG_ENABLE, enable_reg.value)) return false;

  return true;
}
bool AS7341_BankAcessSet(uint8_t RegAdd){

  uint8_t RegLevelNeeded = 0;
  if(RegAdd>=0x60 && RegAdd<=0x74) 
  //if(RegAdd>=0x80) 
    RegLevelNeeded = AS7341_BANK_LOW_ACESS;
  else            
    RegLevelNeeded = AS7341_BANK_HIGH_ACESS;

  if(BankAcessControlValue!=RegLevelNeeded){
    BankAcessControlValue = RegLevelNeeded;
    as7341_cfg0_t cfg0_reg;
    if(!AS7341_read(AS7341_REG_CFG0, &cfg0_reg.value)) return false;

    cfg0_reg.REG_BANK = BankAcessControlValue;
    if(!AS7341_write(AS7341_REG_CFG0, cfg0_reg.value)) return false;
  }
  return true;
}

bool AS7341_SetAcessAndWrite(uint8_t regAdd, uint8_t data){
  AS7341_BankAcessSet(regAdd);
  if(!AS7341_write(regAdd, data)) return false;
  return true;
}
bool AS7341_SetAcessAndRead(uint8_t regAdd, uint8_t *data){
  
  AS7341_BankAcessSet(regAdd);
  if(!AS7341_read(regAdd, data)) return false;
  return true;
}

bool AS7341_SetSMUX(uint8_t* photoDiode, uint8_t* ADC_ID){

  if(!AS7341_DisableSpecMen()) return false;
  // Enable special interrupt and SMUX interrupt
  as7341_cfg9_t cfg9_reg;
  cfg9_reg.value = 0;
  if(!AS7341_SetAcessAndRead(AS7341_REG_CFG9, &cfg9_reg.value)) return false;
  cfg9_reg.SIEN_SMUX = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG9, cfg9_reg.value)) return false;
  
  as7341_intenab_t intenab_reg;
  intenab_reg.value =0;
  if(!AS7341_SetAcessAndRead( AS7341_REG_INTENAB, &intenab_reg.value)) return false;
  intenab_reg.SIEN = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_INTENAB, intenab_reg.value))   return false;
  
  if(!AS7341_WriteI2cReg2SMUX_Sel()) return false;
  if(!AS7341_SetI2cRegSMUX(photoDiode, ADC_ID)) return false;
  
  //UARTprintf("\nSetSMUX\n");
  //CheckArray(photoDiode);
  //UARTprintf("\nADC Config\n");
  //CheckArray(ADC_ID);
  
  //Talvez tenha q ter uma barreira de semafaro aqui
  if(!AS7341_SMUXEnable())           return false;

  return true;
}

bool AS7341_SMUXEnable(){
  as7341_enable_t enable_reg;
  if(!AS7341_SetAcessAndRead(AS7341_REG_ENABLE, &enable_reg.value)) return false;
  enable_reg.SMUXEN = 1;
  if(!AS7341_write(AS7341_REG_ENABLE, enable_reg.value)) return false;

  return true;
}

bool AS7341_WriteI2cReg2SMUX_Sel(){
  as7341_cfg6_t cfg6_reg;
  cfg6_reg.SMUX_CMD = 2;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG6, cfg6_reg.value)) return false;

  return true;
}


bool AS7341_SetI2cRegSMUX(uint8_t* photoDiode, uint8_t* ADC_ID){
  //UARTprintf("\nSet I2C Reg SMUX\n");
  //CheckArray(photoDiode);
  //UARTprintf("\nADC Config\n");
  //CheckArray(ADC_ID);
  uint8_t length = 12;
  uint8_t data2SendAr[12];
  uint8_t data2Send;
  uint8_t RegAdd;
  uint8_t RegAddAr[12];
  // Definições especiais
  /*                      i2c Reg   |  IDs    |   PHOTO  |
                        ------------------------------------
                             0x5     11 e 10     F4 e F2              
                             0xE     29 e 28     F6 e F8             
                            0x10     33 e 32   GPIO e F1               
                            0x11     35 e 34     C2 e INT            
                            0x13     39 e 38   FLKR e NIR            
  */
  
  uint32_t index;
  index = 0;
  as7341_reg1     PixelID_2;
  as7341_reg0     PixelID_1;
  as7341_reg9     PixelID_19;
  as7341_regA     PixelID_20;
  as7341_reg4     PixelID_8;
  as7341_reg3     PixelID_7;
  as7341_regC     PixelID_25;
  as7341_regD     PixelID_26;
  as7341_reg5     PixelID_11;
  as7341_regE     PixelID_28;
  as7341_reg7     PixelID_14;
  as7341_reg6     PixelID_13;
  as7341_regF     PixelID_31;
  as7341_reg0x10  PixelID_32;
  as7341_reg8     PixelID_17;
  as7341_reg0x11  PixelID_35;
  as7341_reg0x13  PixelID_38;
  as7341_reg0x12  PixelID_37;
  
  PixelID_2.value  = 0;
  PixelID_1.value  = 0;
  PixelID_19.value = 0;
  PixelID_20.value = 0;
  PixelID_8.value  = 0;
  PixelID_7.value  = 0;
  PixelID_25.value = 0;
  PixelID_26.value = 0;
  PixelID_11.value = 0;
  PixelID_28.value = 0;
  PixelID_14.value = 0;
  PixelID_13.value = 0;
  PixelID_31.value = 0;
  PixelID_32.value = 0;
  PixelID_17.value = 0;
  PixelID_35.value = 0;
  PixelID_38.value = 0;
  PixelID_37.value = 0;

  while(1){
    
  /*
    UARTprintf("\n\n");
    UARTprintf("ID:  %d\n", photoDiode[index]);
    UARTprintf("ADC: %d\n", ADC_ID[index]);
    UARTprintf("------------\n");
  */  
    if(PHOTO_F1_1   ==photoDiode[index]){
      PixelID_2.MUX_SEL = ADC_ID[index];
      RegAdd = 0x1;
      data2Send = PixelID_2.value;
    } 
    else if(PHOTO_F3_1   ==photoDiode[index]){
      PixelID_1.MUX_SEL = ADC_ID[index];
      RegAdd = 0x0;
      data2Send = PixelID_1.value;
    }
    else if(PHOTO_F5_1   ==photoDiode[index]){
      PixelID_19.MUX_SEL = ADC_ID[index];
      RegAdd = 0x9;
      data2Send = PixelID_19.value;
    }
    else if(PHOTO_F7_1   ==photoDiode[index]){
      PixelID_20.MUX_SEL = ADC_ID[index];
      RegAdd = 0xA;
      data2Send = PixelID_20.value;
    }
    else if(PHOTO_F6_1   ==photoDiode[index]){
      PixelID_8.MUX_SEL = ADC_ID[index];
      RegAdd = 0x4;
      data2Send = PixelID_8.value;
    }
    else if(PHOTO_F8_1   ==photoDiode[index]){
      PixelID_7.MUX_SEL = ADC_ID[index];
      RegAdd = 0x3;
      data2Send = PixelID_7.value;
    }
    else if(PHOTO_F2_1   ==photoDiode[index]){
      PixelID_25.MUX_SEL = ADC_ID[index];
      RegAdd = 0xC;
      data2Send = PixelID_25.value;
    }
    else if(PHOTO_F4_1   ==photoDiode[index]){
      PixelID_26.MUX_SEL = ADC_ID[index];
      RegAdd = 0xD;
      data2Send = PixelID_26.value;
    }
    else if(PHOTO_F4_2   ==(photoDiode[index]& 0xF0)>>4){
      // Escrita Especial
      PixelID_11.value = ADC_ID[index];
      RegAdd = 0x5;
      data2Send = PixelID_11.value;
    }
    //if(PHOTO_F2_2   ==photoDiode[index]){
    //  // Escrita Especial
    //  as7341_reg5 PixelID_10;
    //  PixelID_10.MUX_SEL = ADC_ID[index];
    //  RegAdd = 0x5;
    //  data2Send = PixelID_10.value;
    //}
    else if(PHOTO_F8_2   ==photoDiode[index]){
      // Escrita Especial
      PixelID_28.value = ADC_ID[index];
      RegAdd = 0xE;
      data2Send = PixelID_28.value;
    }
    //if(PHOTO_F6_2   ==photoDiode[index]){
    //  // Escrita Especial
    //  as7341_regE PixelID_29;
    //  PixelID_29.MUX_SEL = ADC_ID[index];
    //  RegAdd = 0xE;
    //  data2Send = PixelID_29.value;
    //}
    else if(PHOTO_F7_2   ==photoDiode[index]){
      PixelID_14.MUX_SEL = ADC_ID[index];
      RegAdd = 0x7;
      data2Send = PixelID_14.value;
    }
    else if(PHOTO_F5_2   ==photoDiode[index]){
      PixelID_13.MUX_SEL = ADC_ID[index];
      RegAdd = 0x6;
      data2Send = PixelID_13.value;
    }
    else if(PHOTO_F3_2   ==photoDiode[index]){
      PixelID_31.MUX_SEL = ADC_ID[index];
      RegAdd = 0xF;
      data2Send = PixelID_31.value;
    }
    else if(PHOTO_F1_2   ==photoDiode[index]){
      // Escrita Especial
      PixelID_32.value = ADC_ID[index];
      RegAdd = 0x10;
      data2Send = PixelID_32.value;
    }
    else if(PHOTO_CLEAR_1==photoDiode[index]){
      PixelID_17.MUX_SEL = ADC_ID[index];
      RegAdd = 0x8;
      data2Send = PixelID_17.value;
    }  
    else if(PHOTO_CLEAR_2==photoDiode[index]){
      // Escrita Especial
      PixelID_35.value = ADC_ID[index];
      RegAdd = 0x11;
      data2Send = PixelID_35.value;
    }  
    else if(PHOTO_NIR    ==photoDiode[index]){
      // Escrita Especial
      PixelID_38.value = ADC_ID[index];
      RegAdd = 0x13;
      data2Send = PixelID_38.value;
    }
    //if(PHOTO_FLICKER==photoDiode[index]){
    //  // Escrita Especial
    //  as7341_reg0x13 PixelID_39;
    //  PixelID_39.MUX_SEL = ADC_ID[index];
    //  RegAdd = 0x13;
    //  data2Send = PixelID_39.value;
    //}   
    //if(GPIO_INPUT   ==photoDiode[index]){
    //  // Escrita Especial
    //  as7341_reg0x10 PixelID_33;
    //  PixelID_33.MUX_SEL = ADC_ID[index];
    //  RegAdd = 0x10;
    //  data2Send = PixelID_33.value;
    //}
    //if(INT_INPUT    ==photoDiode[index]){
    //  // Escrita Especial
    //  as7341_reg0x11 PixelID_34;
    //  PixelID_34.MUX_SEL = ADC_ID[index];
    //  RegAdd = 0x11;
    //  data2Send = PixelID_34.value;
    //}
    else if(DARK         ==photoDiode[index]){
      PixelID_37.MUX_SEL = ADC_ID[index];
      RegAdd = 0x12;
      data2Send = PixelID_37.value;
    }
    
    //data2SendAr[index] = data2Send;
    //RegAddAr[index]    = RegAdd;
    if(!AS7341_write(RegAdd, data2Send)) 
      return false;
    index+=1;
    vTaskDelay(pdMS_TO_TICKS(1));
    if(index==18) 
      break;
  }
/*
   if(!AS7341_write(0x00, 0x30)) return false; // F3 left set to ADC2
   if(!AS7341_write(0x01, 0x01)) return false; // F1 left set to ADC0
   if(!AS7341_write(0x02, 0x00)) return false; // Reserved or disabled
   if(!AS7341_write(0x03, 0x00)) return false; // F8 left disabled
   if(!AS7341_write(0x04, 0x00)) return false; // F6 left disabled
   if(!AS7341_write(0x05, 0x42)) return false; // F4 left connected to ADC3, F2 left connected to ADC1
   if(!AS7341_write(0x06, 0x00)) return false; // F5 left disabled
   if(!AS7341_write(0x07, 0x00)) return false; // F7 left disabled
   if(!AS7341_write(0x08, 0x50)) return false; // CLEAR connected to ADC4
   if(!AS7341_write(0x09, 0x00)) return false; // F5 right disabled
   if(!AS7341_write(0x0A, 0x00)) return false; // F7 right disabled
   if(!AS7341_write(0x0B, 0x00)) return false; // Reserved or disabled
   if(!AS7341_write(0x0C, 0x20)) return false; // F2 right connected to ADC1
   if(!AS7341_write(0x0D, 0x04)) return false; // F4 right connected to ADC3
   if(!AS7341_write(0x0E, 0x00)) return false; // F6/F8 right disabled
   if(!AS7341_write(0x0F, 0x30)) return false; // F3 right connected to ADC2
   if(!AS7341_write(0x10, 0x01)) return false; // F1 right connected to ADC0
   if(!AS7341_write(0x11, 0x50)) return false; // CLEAR right connected to ADC4
   if(!AS7341_write(0x12, 0x00)) return false; // Reserved or disabled
   if(!AS7341_write(0x13, 0x06)) return false; // NIR connected to ADC5
*/
  vTaskDelay(pdMS_TO_TICKS(50));
  //AS7341_writeMultiples()
  return true;
}


bool AS7341_ReadChannels(uint8_t* photoDiode, uint8_t* ADC_config, uint8_t* ADC_count){
  as7341_stat_t stat_rslt;
  as7341_status_t status_rslt;
  as7341_status2_t status2_rslt;
  as7341_status3_t status3_rslt;
  as7341_status5_t status5_rslt;
  as7341_status6_t status6_rslt;
  as7341_astatus_t astat1_rslt;
  as7341_astatus_t astat2_rslt;
  //as7341_intenab_t intenab_reg;
  as7341_control_t control_reg;
  as7341_fifo_lvl_t fifo_lvl;

  uint16_t fifo_buffer;
  control_reg.SP_MAN_AZ       = 0;
  control_reg.FIFO_CLR        = 1; //Talvez valha pena ter uma funcao so pra esse cmd
  control_reg.CLEAR_SAI_ACT   = 0;
  
  float integrationTime = AS7341_GetIntegrationTimeADC()*1000; //tempo em ms
  //if(!AS7341_Enable()) return false;
  
  //UARTprintf("\nReadChannels\n");
  //CheckArray(photoDiode);
  //UARTprintf("\nADC Config\n");
  //CheckArray(ADC_config);
/*
  AS7341_SetAcessAndRead(AS7341_REG_FIFO_LVL, &fifo_lvl.value);
  if(fifo_lvl.value!=0){
    if(!AS7341_SetAcessAndWrite(AS7341_REG_CONTROL, control_reg.value)) return false;
    AS7341_SetAcessAndRead(AS7341_REG_FIFO_LVL, &fifo_lvl.value);
    if(fifo_lvl.value!=0){
      int fifo_index = fifo_lvl.value;
      while(1){
        AS7341_SetAcessAndRead(AS7341_REG_FDATA_L, &fifo_buffer);
        AS7341_SetAcessAndRead(AS7341_REG_FDATA_H, &fifo_buffer);
        fifo_index--;
        if(fifo_index<0) break; 
      }
    }
  }

  AS7341_DeviceStatus(AS7341_REG_STATUS,    &status_rslt.value);
  AS7341_DeviceStatus(AS7341_REG_STATUS5, &status5_rslt.value);
  if(!AS7341_SetAcessAndWrite(AS7341_REG_STATUS, status_rslt.value)) return false;
  AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
  AS7341_SetAcessAndRead(AS7341_REG_FIFO_LVL, &fifo_lvl.value);
*/
  if(!AS7341_SetSMUX(photoDiode, ADC_config)) return false;

  //AS7341_DeviceStatus(AS7341_REG_STATUS,    &status_rslt.value);
  //AS7341_DeviceStatus(AS7341_REG_STATUS5, &status5_rslt.value);
  //if(!AS7341_SetAcessAndWrite(AS7341_REG_STATUS, status_rslt.value)) return false;
  vTaskDelay(pdMS_TO_TICKS(1));
  do{
    AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
    vTaskDelay(pdMS_TO_TICKS(1));
  }while(!stat_rslt.READY);
  //AS7341_DeviceStatus(AS7341_REG_STATUS6, &status6_rslt.value);
//  xSemaphoreTake(AS7341_Semphr, portMAX_DELAY); //SINT_MUX interruption
  
  //AS7341_DeviceStatus(AS7341_REG_STATUS2, &status2_rslt.value);
  //if(!AS7341_SetAcessAndWrite(AS7341_REG_STATUS, status_rslt.value)) return false;
  //do{
  //  AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
  //}while(!stat_rslt.READY);
  if(!AS7341_EnableSpecMen()) return false;
  
  vTaskDelay(pdMS_TO_TICKS((uint32_t)integrationTime));
  do{
    AS7341_DeviceStatus(AS7341_REG_STATUS2, &status2_rslt.value);
  }while(!status2_rslt.AVALID);
  //if(!AS7341_SetAcessAndWrite(AS7341_REG_STATUS, status2_rslt.value)) return false;
  //AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
  
  AS7341_BankAcessSet(AS7341_REG_CH0_DATA_L);
  AS7341_read(AS7341_REG_CH0_DATA_L, ADC_count);
  AS7341_read(AS7341_REG_CH0_DATA_H, ADC_count+1);
  //vTaskDelay(pdMS_TO_TICKS(50));
  if(!AS7341_read(AS7341_REG_CH1_DATA_L, ADC_count+2)) return false;
  if(!AS7341_read(AS7341_REG_CH1_DATA_H, ADC_count+3)) return false;
  //vTaskDelay(pdMS_TO_TICKS(50));
  if(!AS7341_read(AS7341_REG_CH2_DATA_L, ADC_count+4)) return false;
  if(!AS7341_read(AS7341_REG_CH2_DATA_H, ADC_count+5)) return false;
  //vTaskDelay(pdMS_TO_TICKS(50));
  if(!AS7341_read(AS7341_REG_CH3_DATA_L, ADC_count+6)) return false;
  if(!AS7341_read(AS7341_REG_CH3_DATA_H, ADC_count+7)) return false;
  //vTaskDelay(pdMS_TO_TICKS(50));
  if(!AS7341_read(AS7341_REG_CH4_DATA_L, ADC_count+8)) return false;
  if(!AS7341_read(AS7341_REG_CH4_DATA_H, ADC_count+9)) return false;
  //vTaskDelay(pdMS_TO_TICKS(50));
  if(!AS7341_read(AS7341_REG_CH5_DATA_L, ADC_count+10)) return false;
  if(!AS7341_read(AS7341_REG_CH5_DATA_H, ADC_count+11)) return false;
  ////vTaskDelay(pdMS_TO_TICKS(50));
  
  AS7341_SetAcessAndRead(AS7341_REG_FIFO_LVL, &fifo_lvl.value);
  if(!AS7341_DisableSpecMen()) return false; 
  //AS7341_DeviceStatus(AS7341_REG_STATUS5, &status5_rslt.value);
  //AS7341_SpecData(AS7341_REG_ASTATUS2,    &astat2_rslt.value);
  
  //Semafaro que aguarda interrupcao
  //GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_2, GPIO_PIN_2);
  //GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_2, 0);
  AS7341_DeviceStatus(AS7341_REG_STATUS,  &status_rslt.value);
  //AS7341_DeviceStatus(AS7341_REG_STATUS2, &status2_rslt.value);
  //AS7341_DeviceStatus(AS7341_REG_STATUS3, &status3_rslt.value);
  //AS7341_DeviceStatus(AS7341_REG_STATUS5, &status5_rslt.value);
  //AS7341_DeviceStatus(AS7341_REG_STATUS6, &status6_rslt.value);
  //AS7341_SpecData(AS7341_REG_ASTATUS1,    &astat1_rslt.value);

  //control_reg.FIFO_CLR        = 1; //Talvez valha pena ter uma funcao so pra esse cmd
  //if(!AS7341_SetAcessAndWrite(AS7341_REG_CONTROL, control_reg.value)) return false;

  if(!AS7341_SetAcessAndWrite(AS7341_REG_STATUS, status_rslt.value)) return false;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ASTATUS2, astat2_rslt.value)) return false;

  //if(!AS7341_PowerOff()) return false;

  return true;
}

bool AS7341_Boot(){
  if(!AS7341_i2cInit()           ) return false;
  if(!AS7341_PowerOn()            ) return false;
  //if(!AS7341_DisableSpecMen()    ) return false;
  if(!AS7341_DevivceConfig()     ) return false;
  //if(!AS7341_ADC_TimingConfig()  ) return false;
  //if(!AS7341_ADC_Config()        ) return false;     
  //if(!AS7341_InterruptionConfig()) return false;     
  //if(!AS7341_OtherConfig()       ) return false;
  //if(!AS7341_BufferConfig()      ) return false;

  //if(!AS7341_PowerOff()) return false;
  return true;
}

bool AS7341_GetStatus(uint8_t* result){
  as7341_stat_t stat_reg;
  if(!AS7341_SetAcessAndRead(AS7341_REG_STAT, &stat_reg.value)) return false;
  
  *result = stat_reg.value;
  return true;
}
#endif
