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
#include "driverlib/timer.h"
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
#include "DRV8825/drv8825.h"

// Esse codigo se baseia nos exemplos presente em 

// ./Tiva/EK-T4MC/examples/peripherals/i2c/slave_receive_int.c &
// ~/worksapece-v10/hello/external_devices/AS7341_photo.c &
// ./Tiva/EK-TM4C/third_party/FreeRTOS/Demo/CORTEX_LM3S102_Rowley/Demo3/main.c
uint8_t BankAcessControlValue = 0;
uint8_t BankAcessSet =0; // 0 = LowLevel = |0x60 and 0x74|
SemaphoreHandle_t I2C1_Semphr;
SemaphoreHandle_t AS7341_Semphr;
TaskHandle_t xAS7341_DebaunceIntHandle    = NULL;

void PortDIntHanlder();

bool AS7341_PerformanceDbgInit(){
  GPIOUnlockPin(GPIO_PORTB_BASE, GPIO_PIN_3);
  GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, GPIO_PIN_3);
  GPIOPadConfigSet(GPIO_PORTB_BASE, GPIO_PIN_3, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD);
  GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_3, 0);
}
bool AS7341_PerformanceDbgSet(){
  GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_3, GPIO_PIN_3);
}
bool AS7341_PerformanceDbgClr(){
  GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_3, 0);
}

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
  //UARTprintf("\r\t\t\t[PortDIntHandler]\n");
  // Entre o primeiro falling edge até seu rst são necessários
  // 300 us
  // Esse handler é executado em aprox. 5us
  //  A liberação do semaforo ocorre em 5us

  //AS7341_PerformanceDbgSet();    
  
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  GPIOIntClear(GPIO_PORTD_BASE, GPIO_PIN_0);

  vTaskNotifyGiveFromISR(AS7341_Handle, &xHigherPriorityTaskWoken);
  //xSemaphoreGiveFromISR(AS7341_Semphr, &xHigherPriorityTaskWoken);
  
  //AS7341_PerformanceDbgClr();   
  portYIELD();
  //portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void I2C1_IntHandler(){
  I2CMasterIntClear( I2C1_BASE ); 
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  xSemaphoreGiveFromISR(I2C1_Semphr, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

bool WaitACK(){
  // O tempo de agurado de um ACK leva aprox. 74.3us
  //AS7341_PerformanceDbgSet();    
  if(xSemaphoreTake(I2C1_Semphr, pdMS_TO_TICKS(25))==pdFALSE){
    //AS7341_PerformanceDbgClr();   
    return false;
  }
  else{
    //AS7341_PerformanceDbgClr();   
    return true;
  }
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

  //AS7341_PerformanceDbgSet();    
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, false);
  I2CMasterDataPut(I2C1_BASE, regAdd);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_SINGLE_SEND);
  //AS7341_PerformanceDbgClr();   

  //AS7341_PerformanceDbgSet();    
  if(!WaitACK()) return false;
  //AS7341_PerformanceDbgClr();   
  
  //AS7341_PerformanceDbgSet();    
  I2CMasterSlaveAddrSet(I2C1_BASE, AS7341_ADDR, true);
  I2CMasterControl(I2C1_BASE, I2C_MASTER_CMD_SINGLE_RECEIVE);
  //AS7341_PerformanceDbgClr();   

  //AS7341_PerformanceDbgSet();    
  if(!WaitACK()) return false;
  //AS7341_PerformanceDbgClr();   

  //AS7341_PerformanceDbgSet();    
  *data = I2CMasterDataGet(I2C1_BASE);
  //AS7341_PerformanceDbgClr();   
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
  

//  for(index=0; index<length; index++)
//    data_copy[index] = data[index];
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

  I2CMasterInitExpClk(I2C1_BASE, SysCtlClockGet(), true);
  I2CMasterIntEnableEx( I2C1_BASE, I2C_MASTER_INT_DATA );
  IntEnable(INT_I2C1);

  // Enables PD0 to handle AS7341 Interruptions
  AS7341_Semphr = xSemaphoreCreateBinary();

  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
  GPIOUnlockPin(GPIO_PORTD_BASE, GPIO_PIN_0 | GPIO_PIN_1);
  GPIOPinTypeGPIOInput(GPIO_PORTD_BASE, GPIO_PIN_0);
  GPIOPadConfigSet(GPIO_PORTD_BASE, GPIO_PIN_0, GPIO_STRENGTH_8MA, GPIO_PIN_TYPE_STD_WPD);
  GPIOIntTypeSet(GPIO_PORTD_BASE, GPIO_PIN_0, \
                 GPIO_FALLING_EDGE); //|GPIO_LOW_LEVEL);

  GPIOIntClear(GPIO_PORTD_BASE, GPIO_PIN_0);
  GPIOIntEnable(GPIO_PORTD_BASE, GPIO_INT_PIN_0);
  
  IntPrioritySet(INT_GPIOD, 0x7);
  IntRegister(INT_GPIOD,PortDIntHanlder);
  IntEnable(INT_GPIOD);
  IntMasterEnable();

  GPIOPinTypeGPIOOutput(GPIO_PORTD_BASE, GPIO_PIN_1);
  GPIOPinWrite(GPIO_PORTD_BASE, GPIO_PIN_1, GPIO_PIN_1);

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
  as7341_intenab_t intenab_reg;
  intenab_reg.ASIEN   = 0;
  intenab_reg.SP_IEN  = 0;
  intenab_reg.F_IEN   = 0;
  intenab_reg.SIEN    = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_INTENAB, intenab_reg.value)) return false;
  
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
/*
  as7341_cfg1_t cfg1_reg;
  cfg1_reg.AGAIN   = 7; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG1, cfg1_reg.value)) return false;
  
  as7341_cfg10_t cfg10_reg;
  cfg10_reg.AGC_H   = 1; //
  cfg10_reg.AGC_L   = 0; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG10, cfg10_reg.value)) return false;

  as7341_az_config_t az_config_reg;
  az_config_reg.AZ_NTH_ITERATION   = 25; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_AZ_CONFIG, az_config_reg.value)) return false;

  as7341_agc_gain_max_t agc_gain_max_reg;
  agc_gain_max_reg.AGC_AGAIN_MAX = 10;
  agc_gain_max_reg.AGC_FD_GAIN_MAX = 9;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_AGC_GAIN_MAX, agc_gain_max_reg.value)) return false;
*/
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
  return (float)(step*time*2.78f/1000000.0f);
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

uint16_t AS7341_GetGainADC(){
  as7341_cfg1_t cfg1_reg;
  // 1111 1
  //cfg1_reg.AGAIN = value & 0x1F;
  if(!AS7341_SetAcessAndRead(AS7341_REG_CFG1, &cfg1_reg.value)) return false;
  return cfg1_reg.value & 0x1F;
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
  intenab_reg.SIEN   = 1;
  intenab_reg.SP_IEN = 1;
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
    //TaskDelay(pdMS_TO_TICKS(1));
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
  //vTaskDelay(pdMS_TO_TICKS(50));
  //AS7341_writeMultiples()
  return true;
}

bool AS7341_SetSMUXMini(uint8_t* photoDiode, uint8_t* ADC_ID){
  
  as7341_sp_th_l_t lowTh;
  as7341_sp_th_h_t highTh;
  
  lowTh.SP_TH_L_LSB = 0x01;
  lowTh.SP_TH_L_MSB = 0x00;

  highTh.SP_TH_H_LSB = 0x02;
  highTh.SP_TH_H_MSB = 0x00;

  AS7341_SetAcessAndWrite(AS7341_REG_SP_TH_L_LSB, lowTh.SP_TH_L_LSB);
  AS7341_SetAcessAndWrite(AS7341_REG_SP_TH_L_MSB, lowTh.SP_TH_L_MSB);
  AS7341_SetAcessAndWrite(AS7341_REG_SP_TH_H_LSB, highTh.SP_TH_H_LSB);
  AS7341_SetAcessAndWrite(AS7341_REG_SP_TH_H_MSB, highTh.SP_TH_H_MSB);

  float integrationTime = AS7341_GetIntegrationTimeADC()*1000; //tempo em ms
  as7341_stat_t stat_rslt;
  as7341_status2_t status2_rslt;
  as7341_status_t status_rslt;

  UARTprintf("\r[MINI] SetSmux\n");
  if(!AS7341_SetSMUX(photoDiode, ADC_ID)) return false;

  uint32_t readcount1=0;
  uint32_t readcount2=0;
  //xSemaphoreTake(AS7341_Semphr, portMAX_DELAY); //SINT_MUX interruption
  
  AS7341_DeviceStatus(AS7341_REG_STATUS,  &status_rslt.value);
  AS7341_DeviceStatus(AS7341_REG_STATUS2, &status2_rslt.value);
  AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
  if(!stat_rslt.READY){
    do{
      AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
      AS7341_DeviceStatus(AS7341_REG_STATUS2, &status2_rslt.value);
      //vTaskDelay(pdMS_TO_TICKS(1));
      readcount1++;
    }while(!stat_rslt.READY);
  }

  UARTprintf("\r[MINI] Enable\n");
  if(!AS7341_EnableSpecMen()) return false;
  vTaskDelay(pdMS_TO_TICKS((uint32_t)integrationTime));
  
  do{
    AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
    AS7341_DeviceStatus(AS7341_REG_STATUS2, &status2_rslt.value);
    readcount2++;
  }while(!status2_rslt.AVALID);
  UARTprintf("\nS[MUX Mini] Int. Time %d Read Count1 %d Count 2 %d, \n", \
                                      (uint32_t)integrationTime,\
                                      readcount1,\
                                      readcount2);
  
  return true;
}

bool AS7341_ReadChannelsMini(uint8_t* photoDiode, uint8_t* ADC_ID, uint8_t* ADC_count){
  float integrationTime = AS7341_GetIntegrationTimeADC()*1000; //tempo em ms
  uint32_t readcount1=0;
  uint32_t readcount2=0;
  
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

  //xSemaphoreTake(AS7341_Semphr, portMAX_DELAY); //SINT_MUX interruption
  do{
    AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
    vTaskDelay(pdMS_TO_TICKS(1));
    readcount1++;
  }while(!stat_rslt.READY);

  do{
    AS7341_DeviceStatus(AS7341_REG_STATUS2, &status2_rslt.value);
    readcount2++;
  }while(!status2_rslt.AVALID);
  UARTprintf("\n[Read Mini] Int. Time %d Read Count1 %d Count 2 %d, \n", \
                                      (uint32_t)integrationTime,\
                                      readcount1,\
                                      readcount2);
  AS7341_BankAcessSet(AS7341_REG_CH0_DATA_L);
  AS7341_read(AS7341_REG_CH0_DATA_L, ADC_count);
  AS7341_read(AS7341_REG_CH0_DATA_H, ADC_count+1);
  if(!AS7341_read(AS7341_REG_CH1_DATA_L, ADC_count+2)) return false;
  if(!AS7341_read(AS7341_REG_CH1_DATA_H, ADC_count+3)) return false;
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
    AS7341_SetAcessAndRead(AS7341_REG_
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
  //vTaskDelay(pdMS_TO_TICKS(10));

  //AS7341_DeviceStatus(AS7341_REG_STATUS,    &status_rslt.value);
  //AS7341_DeviceStatus(AS7341_REG_STATUS5, &status5_rslt.value);
  //if(!AS7341_SetAcessAndWrite(AS7341_REG_STATUS, status_rslt.value)) return false;

/// * 
  uint32_t readcount1=0;
  uint32_t readcount2=0;
  vTaskDelay(pdMS_TO_TICKS(1));
  do{
    AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
    vTaskDelay(pdMS_TO_TICKS(1));
    readcount1++;
  }while(!stat_rslt.READY);
// * /

  //AS7341_DeviceStatus(AS7341_REG_STATUS6, &status6_rslt.value);
  //xSemaphoreTake(AS7341_Semphr, portMAX_DELAY); //SINT_MUX interruption
  
  //AS7341_DeviceStatus(AS7341_REG_STATUS2, &status2_rslt.value);
  //if(!AS7341_SetAcessAndWrite(AS7341_REG_STATUS, status_rslt.value)) return false;
  //do{
  //  AS7341_DeviceStatus(AS7341_REG_STAT,    &stat_rslt.value);
  //}while(!stat_rslt.READY);
  if(!AS7341_EnableSpecMen()) return false;
  
  vTaskDelay(pdMS_TO_TICKS((uint32_t)integrationTime));
  
  do{
    AS7341_DeviceStatus(AS7341_REG_STATUS2, &status2_rslt.value);
    readcount2++;
  }while(!status2_rslt.AVALID);
  UARTprintf("\nInt. Time %d Read Count1 %d Count 2 %d, \n", \
                                      (uint32_t)integrationTime,\
                                      readcount1,\
                                      readcount2);
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
  ////vTaskDelay(pdMS_TO_TICKS(50));
  if(!AS7341_read(AS7341_REG_CH3_DATA_L, ADC_count+6)) return false;
  if(!AS7341_read(AS7341_REG_CH3_DATA_H, ADC_count+7)) return false;
  ////vTaskDelay(pdMS_TO_TICKS(50));
  if(!AS7341_read(AS7341_REG_CH4_DATA_L, ADC_count+8)) return false;
  if(!AS7341_read(AS7341_REG_CH4_DATA_H, ADC_count+9)) return false;
  ////vTaskDelay(pdMS_TO_TICKS(50));
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
  if(!AS7341_BufferConfig()      ) return false;
    
  //if(!AS7341_PowerOff()) return false;
/*
  xTaskCreate(AS7341_IntWatchDog, "AS7341ReEnInt", configMINIMAL_STACK_SIZE+50, \
                NULL, configMAX_PRIORITIES-8, \
                &xAS7341_DebaunceIntHandle);
*/


  return true;
}

bool AS7341_GetStatus(uint8_t* result){
  as7341_stat_t stat_reg;
  if(!AS7341_SetAcessAndRead(AS7341_REG_STAT, &stat_reg.value)) return false;
  
  *result = stat_reg.value;
  return true;
}

void AS7341_Debaunce(void *ptr) {
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  TimerIntClear(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
  vTaskNotifyGiveFromISR(xAS7341_DebaunceIntHandle, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

bool AS7341_WaitIntSig(){

  if(xSemaphoreTake(AS7341_Semphr, pdMS_TO_TICKS(1500))==pdFALSE) //portMAX_DELAY); //SINT_MUX interruption
    return false;
  else
    return true;
  //AS7341_PerformanceDbgClr();   
}

// USAR M1 PWM4 em PORTF0
bool AS7341_AnalogAproxConfig(uint32_t freq){
  SysCtlPeripheralEnable(SYSCTL_PERIPH_PWM1);
  GPIOPinConfigure(GPIO_PF0_M1PWM4);
  GPIOPinTypePWM(GPIO_PORTF_BASE, GPIO_PIN_0);

  // Configure PWM1 Generator 3
  uint32_t pwmClock = SysCtlClockGet() / 64; // PWM clock is system clock / 64
  uint32_t load = (pwmClock / freq) - 1; // Set the load value based on the frequency
  PWMGenConfigure(PWM1_BASE, PWM_GEN_2, PWM_GEN_MODE_DOWN | PWM_GEN_MODE_NO_SYNC);
  PWMGenPeriodSet(PWM1_BASE, PWM_GEN_2, load);
  
  // Set the PWM duty cycle to 50% initially
  PWMPulseWidthSet(PWM1_BASE, PWM_OUT_4, load / 2);
  // Start PWM generator
  PWMOutputState(PWM1_BASE, PWM_OUT_4_BIT, true);
  PWMGenEnable(PWM1_BASE, PWM_GEN_2);

}
bool AS7341_AnalogAproxDutySet(float PhotoValue){
  float dutyCycle = (PhotoValue/1.8);
  uint32_t load = PWMGenPeriodGet(PWM1_BASE, PWM_GEN_2);
  uint32_t compare = (uint32_t)((1.0f - dutyCycle) * load);

  uint32_t pulseWidth = (uint32_t)((load * dutyCycle));
  PWMPulseWidthSet(PWM1_BASE, PWM_OUT_4, pulseWidth);
  //PWMPulseWidthSet(PWM1_BASE, PWM_OUT_4, compare);

}

const float GeneralSpectralCorrectionMatrix[]={ 0.194140f,  -0.033867f, 0.009500f,  -0.001851f, 0.001581f,  -0.000362f, 0.000774f,  -0.000281f, -0.006954f, -0.000248f,\
  0.196110f,  -0.034209f, 0.009596f,  -0.001870f, 0.001597f,  -0.000366f, 0.000782f,  -0.000284f, -0.007024f, -0.000251f,\ 
  0.198090f,  -0.034555f, 0.009693f,  -0.001889f, 0.001613f,  -0.000370f, 0.000790f,  -0.000287f, -0.007095f, -0.000253f,\
  0.200090f,  -0.034904f, 0.009791f,  -0.001908f, 0.001629f,  -0.000373f, 0.000798f,  -0.000290f, -0.007167f, -0.000256f,\
  0.202110f,  -0.035257f, 0.009890f,  -0.001927f, 0.001646f,  -0.000377f, 0.000806f,  -0.000293f, -0.007239f, -0.000259f,\
  0.204150f,  -0.035613f, 0.009989f,  -0.001947f, 0.001662f,  -0.000381f, 0.000814f,  -0.000296f, -0.007312f, -0.000261f,\
  0.206210f,  -0.035972f, 0.010090f,  -0.001966f, 0.001679f,  -0.000385f, 0.000822f,  -0.000299f, -0.007386f, -0.000264f,\
  0.208290f,  -0.036336f, 0.010192f,  -0.001986f, 0.001696f,  -0.000389f, 0.000831f,  -0.000302f, -0.007461f, -0.000266f,\
  0.210400f,  -0.036703f, 0.010295f,  -0.002006f, 0.001713f,  -0.000393f, 0.000839f,  -0.000305f, -0.007536f, -0.000269f,\
  0.212520f,  -0.037073f, 0.010399f,  -0.002026f, 0.001731f,  -0.000397f, 0.000847f,  -0.000308f, -0.007612f, -0.000272f,\
  0.214670f,  -0.037448f, 0.010504f,  -0.002047f, 0.001748f,  -0.000401f, 0.000856f,  -0.000311f, -0.007689f, -0.000275f,\
  0.216840f,  -0.037826f, 0.010610f,  -0.002068f, 0.001766f,  -0.000405f, 0.000865f,  -0.000314f, -0.007767f, -0.000277f,\
  0.219030f,  -0.038208f, 0.010717f,  -0.002088f, 0.001784f,  -0.000409f, 0.000873f,  -0.000318f, -0.007845f, -0.000280f,\
  0.221240f,  -0.038592f, 0.010825f,  -0.002109f, 0.001802f,  -0.000413f, 0.000882f,  -0.000321f, -0.007924f, -0.000283f,\
  0.223460f,  -0.038975f, 0.010933f,  -0.002130f, 0.001820f,  -0.000417f, 0.000891f,  -0.000324f, -0.008005f, -0.000286f,\
  0.225690f,  -0.039352f, 0.011040f,  -0.002150f, 0.001838f,  -0.000420f, 0.000900f,  -0.000326f, -0.008086f, -0.000288f,\
  0.227920f,  -0.039718f, 0.011145f,  -0.002168f, 0.001856f,  -0.000423f, 0.000909f,  -0.000328f, -0.008168f, -0.000291f,\
  0.230120f,  -0.040061f, 0.011245f,  -0.002185f, 0.001873f,  -0.000425f, 0.000918f,  -0.000329f, -0.008250f, -0.000293f,\
  0.232290f,  -0.040367f, 0.011336f,  -0.002197f, 0.001890f,  -0.000426f, 0.000927f,  -0.000329f, -0.008334f, -0.000296f,\
  0.234380f,  -0.040620f, 0.011415f,  -0.002204f, 0.001905f,  -0.000424f, 0.000936f,  -0.000328f, -0.008418f, -0.000298f,\
  0.236370f,  -0.040797f, 0.011477f,  -0.002204f, 0.001918f,  -0.000420f, 0.000945f,  -0.000325f, -0.008502f, -0.000299f,\
  0.238200f,  -0.040872f, 0.011513f,  -0.002194f, 0.001929f,  -0.000413f, 0.000954f,  -0.000320f, -0.008585f, -0.000300f,\
  0.239840f,  -0.040814f, 0.011517f,  -0.002172f, 0.001935f,  -0.000401f, 0.000963f,  -0.000312f, -0.008667f, -0.000301f,\
  0.241220f,  -0.040592f, 0.011481f,  -0.002133f, 0.001937f,  -0.000384f, 0.000971f,  -0.000302f, -0.008748f, -0.000301f,\
  0.242300f,  -0.040170f, 0.011396f,  -0.002076f, 0.001932f,  -0.000362f, 0.000979f,  -0.000288f, -0.008825f, -0.000301f,\
  0.243010f,  -0.039515f, 0.011253f,  -0.001998f, 0.001920f,  -0.000332f, 0.000986f,  -0.000272f, -0.008899f, -0.000300f,\
  0.243300f,  -0.038591f, 0.011043f,  -0.001894f, 0.001899f,  -0.000295f, 0.000992f,  -0.000251f, -0.008968f, -0.000299f,\
  0.243120f,  -0.037368f, 0.010759f,  -0.001762f, 0.001869f,  -0.000248f, 0.000997f,  -0.000227f, -0.009032f, -0.000297f,\
  0.242410f,  -0.035816f, 0.010394f,  -0.001601f, 0.001828f,  -0.000193f, 0.001002f,  -0.000199f, -0.009089f, -0.000295f,\
  0.241140f,  -0.033911f, 0.009941f,  -0.001407f, 0.001776f,  -0.000127f, 0.001005f,  -0.000166f, -0.009140f, -0.000292f,\
  0.239270f,  -0.031634f, 0.009397f,  -0.001179f, 0.001712f,  -0.000052f, 0.001007f,  -0.000130f, -0.009182f, -0.000289f,\
  0.236770f,  -0.028970f, 0.008759f,  -0.000916f, 0.001636f,  0.000034f,  0.001008f,  -0.000089f, -0.009217f, -0.000286f,\
  0.233640f,  -0.025911f, 0.008025f,  -0.000619f, 0.001547f,  0.000131f,  0.001008f,  -0.000043f, -0.009242f, -0.000282f,\
  0.229850f,  -0.022454f, 0.007196f,  -0.000286f, 0.001446f,  0.000239f,  0.001007f,  0.000006f,  -0.009259f, -0.000277f,\
  0.225410f,  -0.018601f, 0.006274f,  0.000081f,  0.001332f,  0.000356f,  0.001004f,  0.000060f,  -0.009267f, -0.000272f,\
  0.220330f,  -0.014360f, 0.005261f,  0.000480f,  0.001208f,  0.000484f,  0.001000f,  0.000117f,  -0.009265f, -0.000267f,\
  0.214620f,  -0.009745f, 0.004164f,  0.000911f,  0.001072f,  0.000621f,  0.000995f,  0.000179f,  -0.009255f, -0.000262f,\
  0.208300f,  -0.004772f, 0.002989f,  0.001371f,  0.000926f,  0.000767f,  0.000989f,  0.000244f,  -0.009235f, -0.000256f,\
  0.201410f,  0.000537f,  0.001742f,  0.001857f,  0.000772f,  0.000921f,  0.000983f,  0.000313f,  -0.009207f, -0.000249f,\
  0.193970f,  0.006156f,  0.000433f,  0.002365f,  0.000609f,  0.001081f,  0.000975f,  0.000384f,  -0.009171f, -0.000243f,\
  0.186020f,  0.012057f,  -0.000927f, 0.002893f,  0.000441f,  0.001248f,  0.000966f,  0.000459f,  -0.009126f, -0.000236f,\
  0.177610f,  0.018209f,  -0.002329f, 0.003436f,  0.000268f,  0.001419f,  0.000957f,  0.000536f,  -0.009074f, -0.000228f,\
  0.168780f,  0.024577f,  -0.003760f, 0.003990f,  0.000091f,  0.001594f,  0.000948f,  0.000615f,  -0.009016f, -0.000221f,\
  0.159590f,  0.031124f,  -0.005207f, 0.004550f,  -0.000086f, 0.001772f,  0.000938f,  0.000696f,  -0.008951f, -0.000213f,\
  0.150070f,  0.037812f,  -0.006657f, 0.005111f,  -0.000264f, 0.001950f,  0.000928f,  0.000779f,  -0.008880f, -0.000205f,\
  0.140290f,  0.044601f,  -0.008095f, 0.005669f,  -0.000438f, 0.002128f,  0.000918f,  0.000863f,  -0.008803f, -0.000196f,\
  0.130300f,  0.051449f,  -0.009506f, 0.006216f,  -0.000609f, 0.002303f,  0.000909f,  0.000947f,  -0.008722f, -0.000188f,\
  0.120150f,  0.058314f,  -0.010874f, 0.006748f,  -0.000773f, 0.002476f,  0.000899f,  0.001031f,  -0.008636f, -0.000179f,\
  0.109900f,  0.065152f,  -0.012182f, 0.007258f,  -0.000928f, 0.002643f,  0.000890f,  0.001115f,  -0.008547f, -0.000170f,\
  0.099596f,  0.071919f,  -0.013414f, 0.007741f,  -0.001073f, 0.002803f,  0.000882f,  0.001198f,  -0.008453f, -0.000161f,\
  0.089299f,  0.078572f,  -0.014552f, 0.008189f,  -0.001205f, 0.002955f,  0.000875f,  0.001279f,  -0.008357f, -0.000152f,\
  0.079062f,  0.085066f,  -0.015580f, 0.008596f,  -0.001321f, 0.003098f,  0.000869f,  0.001359f,  -0.008257f, -0.000143f,\
  0.068935f,  0.091358f,  -0.016479f, 0.008957f,  -0.001421f, 0.003228f,  0.000864f,  0.001435f,  -0.008155f, -0.000134f,\
  0.058971f,  0.097405f,  -0.017232f, 0.009263f,  -0.001501f, 0.003346f,  0.000861f,  0.001509f,  -0.008051f, -0.000125f,\
  0.049219f,  0.103160f,  -0.017823f, 0.009511f,  -0.001560f, 0.003448f,  0.000859f,  0.001579f,  -0.007945f, -0.000116f,\
  0.039726f,  0.108600f,  -0.018236f, 0.009692f,  -0.001596f, 0.003535f,  0.000859f,  0.001644f,  -0.007837f, -0.000108f,\
  0.030540f,  0.113660f,  -0.018455f, 0.009803f,  -0.001607f, 0.003604f,  0.000860f,  0.001705f,  -0.007727f, -0.000099f,\
  0.021704f,  0.118330f,  -0.018465f, 0.009837f,  -0.001591f, 0.003654f,  0.000864f,  0.001760f,  -0.007616f, -0.000091f,\
  0.013261f,  0.122550f,  -0.018252f, 0.009789f,  -0.001546f, 0.003683f,  0.000870f,  0.001810f,  -0.007504f, -0.000084f,\
  0.005249f,  0.126300f,  -0.017803f, 0.009656f,  -0.001472f, 0.003692f,  0.000877f,  0.001853f,  -0.007390f, -0.000076f,\
  -0.002295f, 0.129550f,  -0.017107f, 0.009432f,  -0.001368f, 0.003678f,  0.000887f,  0.001890f,  -0.007276f, -0.000070f,\
  -0.009336f, 0.132270f,  -0.016154f, 0.009115f,  -0.001231f, 0.003641f,  0.000900f,  0.001919f,  -0.007162f, -0.000063f,\
  -0.015845f, 0.134430f,  -0.014936f, 0.008703f,  -0.001061f, 0.003579f,  0.000914f,  0.001942f,  -0.007046f, -0.000058f,\
  -0.021796f, 0.136020f,  -0.013445f, 0.008192f,  -0.000859f, 0.003494f,  0.000931f,  0.001956f,  -0.006931f, -0.000052f,\
  -0.027166f, 0.137010f,  -0.011676f, 0.007582f,  -0.000622f, 0.003383f,  0.000951f,  0.001963f,  -0.006815f, -0.000048f,\
  -0.031937f, 0.137410f,  -0.009629f, 0.006873f,  -0.000353f, 0.003247f,  0.000972f,  0.001962f,  -0.006700f, -0.000044f,\
  -0.036097f, 0.137190f,  -0.007301f, 0.006065f,  -0.000050f, 0.003087f,  0.000997f,  0.001953f,  -0.006584f, -0.000041f,\
  -0.039639f, 0.136350f,  -0.004697f, 0.005161f,  0.000285f,  0.002902f,  0.001023f,  0.001935f,  -0.006469f, -0.000038f,\
  -0.042560f, 0.134910f,  -0.001822f, 0.004165f,  0.000651f,  0.002692f,  0.001052f,  0.001910f,  -0.006354f, -0.000036f,\
  -0.044868f, 0.132870f,  0.001315f,  0.003079f,  0.001047f,  0.002460f,  0.001082f,  0.001876f,  -0.006239f, -0.000035f,\
  -0.046571f, 0.130240f,  0.004702f,  0.001912f,  0.001470f,  0.002206f,  0.001115f,  0.001835f,  -0.006126f, -0.000035f,\
  -0.047688f, 0.127050f,  0.008323f,  0.000669f,  0.001917f,  0.001931f,  0.001149f,  0.001787f,  -0.006013f, -0.000035f,\
  -0.048242f, 0.123340f,  0.012161f,  -0.000641f, 0.002387f,  0.001639f,  0.001184f,  0.001732f,  -0.005901f, -0.000036f,\
  -0.048261f, 0.119120f,  0.016193f,  -0.002007f, 0.002875f,  0.001329f,  0.001221f,  0.001671f,  -0.005791f, -0.000037f,\
  -0.047780f, 0.114440f,  0.020397f,  -0.003419f, 0.003378f,  0.001006f,  0.001258f,  0.001605f,  -0.005682f, -0.000039f,\
  -0.046835f, 0.109350f,  0.024746f,  -0.004865f, 0.003891f,  0.000672f,  0.001296f,  0.001533f,  -0.005574f, -0.000042f,\
  -0.045467f, 0.103890f,  0.029211f,  -0.006332f, 0.004410f,  0.000329f,  0.001334f,  0.001458f,  -0.005469f, -0.000044f,\
  -0.043721f, 0.098104f,  0.033765f,  -0.007807f, 0.004931f,  -0.000020f, 0.001372f,  0.001379f,  -0.005365f, -0.000048f,\
  -0.041641f, 0.092049f,  0.038377f,  -0.009276f, 0.005448f,  -0.000371f, 0.001409f,  0.001297f,  -0.005263f, -0.000052f,\
  -0.039272f, 0.085774f,  0.043016f,  -0.010724f, 0.005957f,  -0.000722f, 0.001444f,  0.001214f,  -0.005164f, -0.000056f,\
  -0.036661f, 0.079332f,  0.047650f,  -0.012138f, 0.006452f,  -0.001068f, 0.001478f,  0.001130f,  -0.005067f, -0.000060f,\
  -0.033852f, 0.072771f,  0.052249f,  -0.013502f, 0.006928f,  -0.001408f, 0.001510f,  0.001046f,  -0.004972f, -0.000064f,\
  -0.030890f, 0.066144f,  0.056783f,  -0.014802f, 0.007380f,  -0.001737f, 0.001539f,  0.000962f,  -0.004881f, -0.000069f,\
  -0.027817f, 0.059497f,  0.061220f,  -0.016024f, 0.007803f,  -0.002051f, 0.001565f,  0.000881f,  -0.004792f, -0.000074f,\
  -0.024673f, 0.052876f,  0.065532f,  -0.017152f, 0.008189f,  -0.002349f, 0.001587f,  0.000801f,  -0.004706f, -0.000079f,\
  -0.021495f, 0.046325f,  0.069690f,  -0.018174f, 0.008536f,  -0.002626f, 0.001605f,  0.000725f,  -0.004624f, -0.000084f,\
  -0.018318f, 0.039885f,  0.073668f,  -0.019074f, 0.008836f,  -0.002879f, 0.001617f,  0.000653f,  -0.004545f, -0.000090f,\
  -0.015173f, 0.033592f,  0.077439f,  -0.019840f, 0.009084f,  -0.003105f, 0.001625f,  0.000585f,  -0.004470f, -0.000095f,\
  -0.012088f, 0.027481f,  0.080980f,  -0.020459f, 0.009276f,  -0.003301f, 0.001626f,  0.000522f,  -0.004398f, -0.000100f,\
  -0.009087f, 0.021582f,  0.084267f,  -0.020920f, 0.009406f,  -0.003463f, 0.001620f,  0.000465f,  -0.004331f, -0.000106f,\
  -0.006192f, 0.015923f,  0.087282f,  -0.021210f, 0.009470f,  -0.003590f, 0.001607f,  0.000414f,  -0.004267f, -0.000112f,\
  -0.003420f, 0.010526f,  0.090004f,  -0.021319f, 0.009463f,  -0.003679f, 0.001586f,  0.000370f,  -0.004207f, -0.000117f,\
  -0.000787f, 0.005413f,  0.092418f,  -0.021238f, 0.009381f,  -0.003727f, 0.001557f,  0.000333f,  -0.004151f, -0.000123f,\
  0.001696f,  0.000600f,  0.094510f,  -0.020960f, 0.009221f,  -0.003732f, 0.001519f,  0.000303f,  -0.004100f, -0.000130f,\
  0.004018f,  -0.003899f, 0.096267f,  -0.020476f, 0.008980f,  -0.003693f, 0.001471f,  0.000281f,  -0.004052f, -0.000136f,\
  0.006173f,  -0.008073f, 0.097679f,  -0.019782f, 0.008657f,  -0.003608f, 0.001415f,  0.000267f,  -0.004009f, -0.000143f,\
  0.008156f,  -0.011913f, 0.098738f,  -0.018874f, 0.008248f,  -0.003478f, 0.001348f,  0.000261f,  -0.003970f, -0.000149f,\
  0.009963f,  -0.015415f, 0.099441f,  -0.017750f, 0.007755f,  -0.003300f, 0.001272f,  0.000263f,  -0.003935f, -0.000157f,\
  0.011593f,  -0.018575f, 0.099783f,  -0.016408f, 0.007176f,  -0.003076f, 0.001186f,  0.000273f,  -0.003905f, -0.000164f,\
  0.013046f,  -0.021392f, 0.099764f,  -0.014850f, 0.006514f,  -0.002806f, 0.001091f,  0.000290f,  -0.003879f, -0.000172f,\
  0.014322f,  -0.023867f, 0.099387f,  -0.013079f, 0.005770f,  -0.002491f, 0.000986f,  0.000316f,  -0.003856f, -0.000180f,\
  0.015423f,  -0.026003f, 0.098654f,  -0.011099f, 0.004947f,  -0.002133f, 0.000872f,  0.000349f,  -0.003838f, -0.000188f,\
  0.016353f,  -0.027806f, 0.097574f,  -0.008915f, 0.004050f,  -0.001734f, 0.000750f,  0.000389f,  -0.003823f, -0.000196f,\
  0.017116f,  -0.029282f, 0.096154f,  -0.006537f, 0.003082f,  -0.001295f, 0.000620f,  0.000436f,  -0.003812f, -0.000205f,\
  0.017716f,  -0.030440f, 0.094405f,  -0.003974f, 0.002050f,  -0.000822f, 0.000484f,  0.000490f,  -0.003804f, -0.000214f,\
  0.018157f,  -0.031289f, 0.092339f,  -0.001235f, 0.000960f,  -0.000316f, 0.000341f,  0.000549f,  -0.003800f, -0.000223f,\
  0.018448f,  -0.031840f, 0.089972f,  0.001665f,  -0.000182f, 0.000219f,  0.000193f,  0.000614f,  -0.003799f, -0.000232f,\
  0.018592f,  -0.032107f, 0.087319f,  0.004714f,  -0.001366f, 0.000777f,  0.000040f,  0.000684f,  -0.003800f, -0.000242f,\
  0.018599f,  -0.032102f, 0.084398f,  0.007897f,  -0.002584f, 0.001356f,  -0.000115f, 0.000758f,  -0.003804f, -0.000251f,\
  0.018474f,  -0.031841f, 0.081227f,  0.011198f,  -0.003829f, 0.001949f,  -0.000272f, 0.000835f,  -0.003810f, -0.000261f,\
  0.018225f,  -0.031338f, 0.077826f,  0.014601f,  -0.005090f, 0.002552f,  -0.000429f, 0.000915f,  -0.003818f, -0.000270f,\
  0.017862f,  -0.030610f, 0.074217f,  0.018088f,  -0.006357f, 0.003160f,  -0.000586f, 0.000998f,  -0.003828f, -0.000279f,\
  0.017391f,  -0.029673f, 0.070420f,  0.021642f,  -0.007621f, 0.003768f,  -0.000740f, 0.001082f,  -0.003839f, -0.000288f,\
  0.016821f,  -0.028545f, 0.066459f,  0.025243f,  -0.008872f, 0.004370f,  -0.000892f, 0.001167f,  -0.003851f, -0.000297f,\
  0.016161f,  -0.027242f, 0.062355f,  0.028874f,  -0.010099f, 0.004961f,  -0.001038f, 0.001251f,  -0.003864f, -0.000305f,\
  0.015420f,  -0.025784f, 0.058133f,  0.032514f,  -0.011293f, 0.005537f,  -0.001179f, 0.001335f,  -0.003878f, -0.000313f,\
  0.014607f,  -0.024188f, 0.053815f,  0.036146f,  -0.012443f, 0.006091f,  -0.001313f, 0.001417f,  -0.003891f, -0.000321f,\
  0.013731f,  -0.022473f, 0.049426f,  0.039750f,  -0.013539f, 0.006620f,  -0.001439f, 0.001496f,  -0.003904f, -0.000328f,\
  0.012800f,  -0.020656f, 0.044988f,  0.043307f,  -0.014571f, 0.007118f,  -0.001555f, 0.001573f,  -0.003917f, -0.000335f,\
  0.011825f,  -0.018756f, 0.040526f,  0.046798f,  -0.015531f, 0.007581f,  -0.001661f, 0.001646f,  -0.003928f, -0.000341f,\
  0.010814f,  -0.016791f, 0.036062f,  0.050205f,  -0.016409f, 0.008004f,  -0.001755f, 0.001715f,  -0.003938f, -0.000347f,\
  0.009775f,  -0.014778f, 0.031618f,  0.053509f,  -0.017196f, 0.008384f,  -0.001838f, 0.001778f,  -0.003947f, -0.000352f,\
  0.008718f,  -0.012735f, 0.027217f,  0.056693f,  -0.017883f, 0.008717f,  -0.001907f, 0.001836f,  -0.003954f, -0.000356f,\
  0.007652f,  -0.010678f, 0.022881f,  0.059740f,  -0.018464f, 0.008999f,  -0.001963f, 0.001887f,  -0.003958f, -0.000359f,\
  0.006584f,  -0.008625f, 0.018630f,  0.062634f,  -0.018930f, 0.009227f,  -0.002004f, 0.001931f,  -0.003960f, -0.000362f,\
  0.005524f,  -0.006590f, 0.014483f,  0.065358f,  -0.019273f, 0.009398f,  -0.002031f, 0.001968f,  -0.003959f, -0.000364f,\
  0.004478f,  -0.004589f, 0.010460f,  0.067898f,  -0.019488f, 0.009509f,  -0.002042f, 0.001997f,  -0.003955f, -0.000366f,\
  0.003453f,  -0.002635f, 0.006579f,  0.070239f,  -0.019567f, 0.009557f,  -0.002036f, 0.002017f,  -0.003947f, -0.000367f,\
  0.002458f,  -0.000742f, 0.002855f,  0.072368f,  -0.019504f, 0.009539f,  -0.002014f, 0.002029f,  -0.003936f, -0.000366f,\
  0.001498f,  0.001078f,  -0.000695f, 0.074272f,  -0.019293f, 0.009453f,  -0.001975f, 0.002030f,  -0.003920f, -0.000366f,\
  0.000579f,  0.002813f,  -0.004058f, 0.075941f,  -0.018927f, 0.009295f,  -0.001918f, 0.002022f,  -0.003901f, -0.000364f,\
  -0.000293f, 0.004455f,  -0.007222f, 0.077363f,  -0.018403f, 0.009064f,  -0.001842f, 0.002003f,  -0.003877f, -0.000361f,\
  -0.001115f, 0.005993f,  -0.010174f, 0.078530f,  -0.017714f, 0.008755f,  -0.001747f, 0.001973f,  -0.003848f, -0.000358f,\
  -0.001882f, 0.007421f,  -0.012906f, 0.079434f,  -0.016857f, 0.008368f,  -0.001632f, 0.001932f,  -0.003814f, -0.000354f,\
  -0.002592f, 0.008732f,  -0.015409f, 0.080067f,  -0.015826f, 0.007898f,  -0.001496f, 0.001879f,  -0.003776f, -0.000349f,\
  -0.003241f, 0.009921f,  -0.017678f, 0.080425f,  -0.014619f, 0.007346f,  -0.001339f, 0.001814f,  -0.003732f, -0.000343f,\
  -0.003829f, 0.010985f,  -0.019707f, 0.080503f,  -0.013233f, 0.006708f,  -0.001161f, 0.001738f,  -0.003683f, -0.000336f,\
  -0.004353f, 0.011920f,  -0.021494f, 0.080301f,  -0.011667f, 0.005984f,  -0.000960f, 0.001649f,  -0.003630f, -0.000328f,\
  -0.004813f, 0.012725f,  -0.023038f, 0.079818f,  -0.009921f, 0.005173f,  -0.000737f, 0.001548f,  -0.003571f, -0.000319f,\
  -0.005210f, 0.013401f,  -0.024340f, 0.079055f,  -0.007996f, 0.004277f,  -0.000492f, 0.001435f,  -0.003507f, -0.000309f,\
  -0.005544f, 0.013949f,  -0.025402f, 0.078017f,  -0.005894f, 0.003296f,  -0.000226f, 0.001311f,  -0.003439f, -0.000298f,\
  -0.005817f, 0.014370f,  -0.026227f, 0.076708f,  -0.003621f, 0.002233f,  0.000062f,  0.001175f,  -0.003366f, -0.000286f,\
  -0.006029f, 0.014667f,  -0.026821f, 0.075138f,  -0.001182f, 0.001092f,  0.000370f,  0.001028f,  -0.003289f, -0.000273f,\
  -0.006183f, 0.014845f,  -0.027192f, 0.073314f,  0.001413f,  -0.000122f, 0.000697f,  0.000872f,  -0.003208f, -0.000260f,\
  -0.006280f, 0.014908f,  -0.027348f, 0.071250f,  0.004156f,  -0.001403f, 0.001040f,  0.000706f,  -0.003123f, -0.000245f,\
  -0.006324f, 0.014863f,  -0.027299f, 0.068959f,  0.007033f,  -0.002741f, 0.001399f,  0.000533f,  -0.003036f, -0.000229f,\
  -0.006318f, 0.014714f,  -0.027056f, 0.066456f,  0.010030f,  -0.004129f, 0.001769f,  0.000352f,  -0.002946f, -0.000212f,\
  -0.006263f, 0.014469f,  -0.026630f, 0.063758f,  0.013132f,  -0.005556f, 0.002148f,  0.000167f,  -0.002854f, -0.000195f,\
  -0.006164f, 0.014135f,  -0.026036f, 0.060884f,  0.016322f,  -0.007011f, 0.002533f,  -0.000023f, -0.002761f, -0.000177f,\
  -0.006022f, 0.013719f,  -0.025286f, 0.057853f,  0.019580f,  -0.008480f, 0.002920f,  -0.000215f, -0.002667f, -0.000158f,\
  -0.005842f, 0.013229f,  -0.024395f, 0.054685f,  0.022886f,  -0.009952f, 0.003306f,  -0.000408f, -0.002573f, -0.000139f,\
  -0.005627f, 0.012672f,  -0.023378f, 0.051402f,  0.026221f,  -0.011411f, 0.003686f,  -0.000600f, -0.002480f, -0.000120f,\
  -0.005380f, 0.012058f,  -0.022248f, 0.048026f,  0.029563f,  -0.012845f, 0.004056f,  -0.000789f, -0.002388f, -0.000100f,\
  -0.005103f, 0.011393f,  -0.021021f, 0.044577f,  0.032891f,  -0.014238f, 0.004413f,  -0.000974f, -0.002298f, -0.000080f,\
  -0.004802f, 0.010685f,  -0.019711f, 0.041077f,  0.036183f,  -0.015577f, 0.004752f,  -0.001153f, -0.002210f, -0.000061f,\
  -0.004477f, 0.009942f,  -0.018333f, 0.037548f,  0.039419f,  -0.016849f, 0.005069f,  -0.001324f, -0.002126f, -0.000041f,\
  -0.004134f, 0.009171f,  -0.016901f, 0.034010f,  0.042578f,  -0.018039f, 0.005361f,  -0.001485f, -0.002046f, -0.000022f,\
  -0.003774f, 0.008380f,  -0.015428f, 0.030483f,  0.045642f,  -0.019135f, 0.005624f,  -0.001636f, -0.001970f, -0.000003f,\
  -0.003401f, 0.007575f,  -0.013927f, 0.026986f,  0.048591f,  -0.020125f, 0.005855f,  -0.001775f, -0.001898f, 0.000016f,
  -0.003017f, 0.006762f,  -0.012410f, 0.023535f,  0.051408f,  -0.020999f, 0.006051f,  -0.001901f, -0.001832f, 0.000033f,
  -0.002627f, 0.005949f,  -0.010888f, 0.020149f,  0.054078f,  -0.021747f, 0.006209f,  -0.002012f, -0.001771f, 0.000050f,
  -0.002232f, 0.005140f,  -0.009373f, 0.016842f,  0.056585f,  -0.022358f, 0.006327f,  -0.002108f, -0.001717f, 0.000066f,
  -0.001835f, 0.004341f,  -0.007874f, 0.013629f,  0.058917f,  -0.022827f, 0.006404f,  -0.002188f, -0.001668f, 0.000081f,
  -0.001438f, 0.003557f,  -0.006401f, 0.010522f,  0.061061f,  -0.023145f, 0.006437f,  -0.002251f, -0.001626f, 0.000095f,
  -0.001045f, 0.002793f,  -0.004961f, 0.007534f,  0.063007f,  -0.023308f, 0.006426f,  -0.002297f, -0.001591f, 0.000108f,
  -0.000658f, 0.002052f,  -0.003562f, 0.004674f,  0.064746f,  -0.023310f, 0.006369f,  -0.002325f, -0.001562f, 0.000119f,
  -0.000279f, 0.001339f,  -0.002212f, 0.001952f,  0.066270f,  -0.023148f, 0.006266f,  -0.002336f, -0.001540f, 0.000129f,
  0.000091f,  0.000657f,  -0.000917f, -0.000624f, 0.067573f,  -0.022820f, 0.006118f,  -0.002328f, -0.001524f, 0.000138f,
  0.000449f,  0.000009f,  0.000319f,  -0.003047f, 0.068649f,  -0.022323f, 0.005923f,  -0.002303f, -0.001516f, 0.000145f,
  0.000792f,  -0.000602f, 0.001490f,  -0.005311f, 0.069496f,  -0.021657f, 0.005683f,  -0.002259f, -0.001514f, 0.000151f,
  0.001121f,  -0.001174f, 0.002592f,  -0.007411f, 0.070109f,  -0.020823f, 0.005398f,  -0.002199f, -0.001518f, 0.000155f,
  0.001432f,  -0.001705f, 0.003622f,  -0.009343f, 0.070488f,  -0.019822f, 0.005069f,  -0.002120f, -0.001529f, 0.000157f,
  0.001725f,  -0.002194f, 0.004576f,  -0.011106f, 0.070632f,  -0.018655f, 0.004698f,  -0.002026f, -0.001546f, 0.000158f,
  0.001998f,  -0.002638f, 0.005453f,  -0.012695f, 0.070541f,  -0.017325f, 0.004287f,  -0.001915f, -0.001569f, 0.000157f,
  0.002250f,  -0.003037f, 0.006251f,  -0.014111f, 0.070218f,  -0.015837f, 0.003836f,  -0.001788f, -0.001598f, 0.000155f,
  0.002480f,  -0.003391f, 0.006968f,  -0.015354f, 0.069664f,  -0.014194f, 0.003349f,  -0.001647f, -0.001633f, 0.000150f,
  0.002688f,  -0.003698f, 0.007603f,  -0.016423f, 0.068885f,  -0.012401f, 0.002827f,  -0.001492f, -0.001672f, 0.000145f,
  0.002872f,  -0.003959f, 0.008157f,  -0.017321f, 0.067883f,  -0.010465f, 0.002274f,  -0.001324f, -0.001716f, 0.000137f,
  0.003032f,  -0.004173f, 0.008629f,  -0.018049f, 0.066665f,  -0.008393f, 0.001691f,  -0.001144f, -0.001765f, 0.000127f,
  0.003168f,  -0.004341f, 0.009020f,  -0.018610f, 0.065238f,  -0.006190f, 0.001083f,  -0.000953f, -0.001818f, 0.000116f,
  0.003278f,  -0.004464f, 0.009331f,  -0.019008f, 0.063608f,  -0.003866f, 0.000451f,  -0.000753f, -0.001875f, 0.000104f,
  0.003365f,  -0.004543f, 0.009564f,  -0.019247f, 0.061783f,  -0.001429f, -0.000199f, -0.000544f, -0.001934f, 0.000089f,
  0.003426f,  -0.004578f, 0.009720f,  -0.019332f, 0.059773f,  0.001112f,  -0.000865f, -0.000327f, -0.001997f, 0.000074f,
  0.003462f,  -0.004570f, 0.009801f,  -0.019268f, 0.057587f,  0.003747f,  -0.001543f, -0.000105f, -0.002062f, 0.000056f,
  0.003474f,  -0.004522f, 0.009810f,  -0.019062f, 0.055235f,  0.006465f,  -0.002229f, 0.000121f,  -0.002130f, 0.000037f,
  0.003462f,  -0.004435f, 0.009750f,  -0.018720f, 0.052729f,  0.009257f,  -0.002919f, 0.000350f,  -0.002198f, 0.000017f,
  0.003426f,  -0.004310f, 0.009624f,  -0.018249f, 0.050081f,  0.012110f,  -0.003607f, 0.000581f,  -0.002268f, -0.000004f,\
  0.003368f,  -0.004151f, 0.009434f,  -0.017656f, 0.047301f,  0.015012f,  -0.004291f, 0.000812f,  -0.002338f, -0.000027f,\
  0.003287f,  -0.003958f, 0.009185f,  -0.016950f, 0.044404f,  0.017953f,  -0.004965f, 0.001041f,  -0.002409f, -0.000051f,\
  0.003185f,  -0.003734f, 0.008880f,  -0.016138f, 0.041403f,  0.020918f,  -0.005624f, 0.001266f,  -0.002478f, -0.000076f,\
  0.003063f,  -0.003482f, 0.008524f,  -0.015230f, 0.038311f,  0.023895f,  -0.006264f, 0.001487f,  -0.002547f, -0.000102f,\
  0.002922f,  -0.003205f, 0.008119f,  -0.014234f, 0.035143f,  0.026870f,  -0.006880f, 0.001701f,  -0.002615f, -0.000129f,\
  0.002762f,  -0.002904f, 0.007672f,  -0.013160f, 0.031913f,  0.029831f,  -0.007467f, 0.001906f,  -0.002680f, -0.000157f,\
  0.002587f,  -0.002583f, 0.007186f,  -0.012017f, 0.028637f,  0.032763f,  -0.008020f, 0.002102f,  -0.002743f, -0.000185f,\
  0.002396f,  -0.002244f, 0.006665f,  -0.010814f, 0.025329f,  0.035653f,  -0.008534f, 0.002285f,  -0.002804f, -0.000214f,\
  0.002192f,  -0.001892f, 0.006115f,  -0.009563f, 0.022005f,  0.038486f,  -0.009003f, 0.002456f,  -0.002860f, -0.000243f,\
  0.001976f,  -0.001527f, 0.005540f,  -0.008271f, 0.018680f,  0.041249f,  -0.009424f, 0.002611f,  -0.002913f, -0.000273f,\
  0.001750f,  -0.001155f, 0.004944f,  -0.006951f, 0.015371f,  0.043927f,  -0.009790f, 0.002749f,  -0.002962f, -0.000303f,\
  0.001516f,  -0.000777f, 0.004334f,  -0.005611f, 0.012093f,  0.046505f,  -0.010097f, 0.002869f,  -0.003006f, -0.000333f,\
  0.001275f,  -0.000397f, 0.003713f,  -0.004262f, 0.008861f,  0.048971f,  -0.010339f, 0.002969f,  -0.003044f, -0.000363f,\
  0.001030f,  -0.000018f, 0.003086f,  -0.002913f, 0.005692f,  0.051310f,  -0.010513f, 0.003047f,  -0.003077f, -0.000392f,\
  0.000782f,  0.000357f,  0.002458f,  -0.001576f, 0.002601f,  0.053509f,  -0.010613f, 0.003103f,  -0.003104f, -0.000421f,\
  0.000533f,  0.000724f,  0.001834f,  -0.000259f, -0.000398f, 0.055554f,  -0.010636f, 0.003133f,  -0.003125f, -0.000450f,\
  0.000286f,  0.001082f,  0.001219f,  0.001028f,  -0.003289f, 0.057433f,  -0.010576f, 0.003139f,  -0.003139f, -0.000478f,\
  0.000041f,  0.001426f,  0.000616f,  0.002275f,  -0.006058f, 0.059134f,  -0.010430f, 0.003117f,  -0.003145f, -0.000506f,\
  -0.000198f, 0.001754f,  0.000030f,  0.003475f,  -0.008692f, 0.060646f,  -0.010194f, 0.003067f,  -0.003145f, -0.000532f,\
  -0.000430f, 0.002064f,  -0.000534f, 0.004617f,  -0.011177f, 0.061957f,  -0.009866f, 0.002988f,  -0.003136f, -0.000558f,\
  -0.000654f, 0.002352f,  -0.001073f, 0.005695f,  -0.013502f, 0.063058f,  -0.009442f, 0.002879f,  -0.003120f, -0.000582f,\
  -0.000867f, 0.002617f,  -0.001583f, 0.006701f,  -0.015657f, 0.063941f,  -0.008920f, 0.002741f,  -0.003097f, -0.000605f,\
  -0.001068f, 0.002857f,  -0.002062f, 0.007629f,  -0.017631f, 0.064598f,  -0.008298f, 0.002571f,  -0.003065f, -0.000627f,\
  -0.001256f, 0.003069f,  -0.002505f, 0.008472f,  -0.019416f, 0.065025f,  -0.007576f, 0.002371f,  -0.003025f, -0.000647f,\
  -0.001430f, 0.003253f,  -0.002911f, 0.009228f,  -0.021007f, 0.065218f,  -0.006753f, 0.002140f,  -0.002977f, -0.000666f,\
  -0.001589f, 0.003407f,  -0.003278f, 0.009891f,  -0.022398f, 0.065175f,  -0.005831f, 0.001880f,  -0.002921f, -0.000683f,\
  -0.001731f, 0.003531f,  -0.003605f, 0.010459f,  -0.023588f, 0.064897f,  -0.004810f, 0.001590f,  -0.002858f, -0.000699f,\
  -0.001858f, 0.003625f,  -0.003890f, 0.010932f,  -0.024577f, 0.064386f,  -0.003694f, 0.001272f,  -0.002787f, -0.000712f,\
  -0.001968f, 0.003688f,  -0.004133f, 0.011309f,  -0.025365f, 0.063647f,  -0.002486f, 0.000928f,  -0.002709f, -0.000724f,\
  -0.002063f, 0.003722f,  -0.004334f, 0.011591f,  -0.025957f, 0.062687f,  -0.001191f, 0.000559f,  -0.002624f, -0.000734f,\
  -0.002141f, 0.003727f,  -0.004495f, 0.011781f,  -0.026359f, 0.061515f,  0.000186f,  0.000168f,  -0.002534f, -0.000743f,\
  -0.002205f, 0.003705f,  -0.004615f, 0.011883f,  -0.026577f, 0.060141f,  0.001640f,  -0.000243f, -0.002438f, -0.000750f,\
  -0.002255f, 0.003658f,  -0.004697f, 0.011898f,  -0.026619f, 0.058577f,  0.003164f,  -0.000672f, -0.002337f, -0.000754f,\
  -0.002290f, 0.003586f,  -0.004743f, 0.011833f,  -0.026495f, 0.056835f,  0.004750f,  -0.001116f, -0.002232f, -0.000758f,\
  -0.002314f, 0.003492f,  -0.004753f, 0.011691f,  -0.026215f, 0.054927f,  0.006391f,  -0.001571f, -0.002123f, -0.000759f,\
  -0.002325f, 0.003377f,  -0.004729f, 0.011478f,  -0.025788f, 0.052869f,  0.008081f,  -0.002035f, -0.002011f, -0.000759f,\
  -0.002326f, 0.003245f,  -0.004675f, 0.011199f,  -0.025225f, 0.050673f,  0.009812f,  -0.002505f, -0.001897f, -0.000757f,\
  -0.002317f, 0.003095f,  -0.004592f, 0.010859f,  -0.024538f, 0.048355f,  0.011575f,  -0.002979f, -0.001782f, -0.000754f,\
  -0.002299f, 0.002931f,  -0.004482f, 0.010465f,  -0.023736f, 0.045927f,  0.013363f,  -0.003452f, -0.001666f, -0.000749f,\
  -0.002272f, 0.002755f,  -0.004347f, 0.010020f,  -0.022831f, 0.043405f,  0.015169f,  -0.003922f, -0.001550f, -0.000742f,\
  -0.002238f, 0.002568f,  -0.004191f, 0.009531f,  -0.021834f, 0.040803f,  0.016984f,  -0.004386f, -0.001435f, -0.000734f,\
  -0.002197f, 0.002372f,  -0.004014f, 0.009004f,  -0.020754f, 0.038134f,  0.018802f,  -0.004841f, -0.001321f, -0.000724f,\
  -0.002149f, 0.002169f,  -0.003819f, 0.008443f,  -0.019604f, 0.035413f,  0.020614f,  -0.005284f, -0.001210f, -0.000713f,\
  -0.002096f, 0.001962f,  -0.003608f, 0.007854f,  -0.018392f, 0.032652f,  0.022414f,  -0.005712f, -0.001101f, -0.000701f,\
  -0.002037f, 0.001750f,  -0.003384f, 0.007243f,  -0.017129f, 0.029865f,  0.024193f,  -0.006122f, -0.000996f, -0.000687f,\
  -0.001974f, 0.001537f,  -0.003148f, 0.006613f,  -0.015825f, 0.027066f,  0.025946f,  -0.006512f, -0.000896f, -0.000671f,\
  -0.001905f, 0.001324f,  -0.002903f, 0.005970f,  -0.014489f, 0.024265f,  0.027665f,  -0.006878f, -0.000800f, -0.000655f,\
  -0.001833f, 0.001112f,  -0.002650f, 0.005320f,  -0.013130f, 0.021476f,  0.029343f,  -0.007219f, -0.000709f, -0.000637f,\
  -0.001756f, 0.000902f,  -0.002391f, 0.004665f,  -0.011757f, 0.018710f,  0.030974f,  -0.007531f, -0.000625f, -0.000617f,\
  -0.001674f, 0.000695f,  -0.002129f, 0.004010f,  -0.010378f, 0.015977f,  0.032552f,  -0.007813f, -0.000547f, -0.000597f,\
  -0.001589f, 0.000494f,  -0.001864f, 0.003360f,  -0.009000f, 0.013289f,  0.034072f,  -0.008061f, -0.000477f, -0.000575f,\
  -0.001500f, 0.000298f,  -0.001598f, 0.002717f,  -0.007631f, 0.010653f,  0.035527f,  -0.008275f, -0.000414f, -0.000552f,\
  -0.001406f, 0.000108f,  -0.001332f, 0.002086f,  -0.006277f, 0.008080f,  0.036913f,  -0.008452f, -0.000358f, -0.000528f,\
  -0.001308f, -0.000074f, -0.001068f, 0.001469f,  -0.004945f, 0.005578f,  0.038224f,  -0.008590f, -0.000312f, -0.000503f,\
  -0.001206f, -0.000249f, -0.000807f, 0.000869f,  -0.003640f, 0.003154f,  0.039458f,  -0.008687f, -0.000273f, -0.000477f,\
  -0.001099f, -0.000415f, -0.000550f, 0.000288f,  -0.002366f, 0.000815f,  0.040609f,  -0.008743f, -0.000244f, -0.000449f,\
  -0.000987f, -0.000572f, -0.000298f, -0.000270f, -0.001130f, -0.001434f, 0.041673f,  -0.008756f, -0.000223f, -0.000421f,\
  -0.000871f, -0.000721f, -0.000051f, -0.000805f, 0.000066f,  -0.003586f, 0.042649f,  -0.008724f, -0.000212f, -0.000392f,\
  -0.000750f, -0.000860f, 0.000189f,  -0.001315f, 0.001218f,  -0.005636f, 0.043533f,  -0.008647f, -0.000210f, -0.000362f,\
  -0.000625f, -0.000990f, 0.000422f,  -0.001798f, 0.002323f,  -0.007581f, 0.044322f,  -0.008524f, -0.000218f, -0.000331f,\
  -0.000495f, -0.001109f, 0.000648f,  -0.002252f, 0.003378f,  -0.009417f, 0.045014f,  -0.008354f, -0.000235f, -0.000299f,\
  -0.000362f, -0.001217f, 0.000866f,  -0.002677f, 0.004380f,  -0.011141f, 0.045609f,  -0.008136f, -0.000262f, -0.000266f,\
  -0.000225f, -0.001314f, 0.001075f,  -0.003071f, 0.005327f,  -0.012750f, 0.046103f,  -0.007871f, -0.000299f, -0.000233f,\
  -0.000087f, -0.001400f, 0.001274f,  -0.003433f, 0.006218f,  -0.014242f, 0.046498f,  -0.007557f, -0.000345f, -0.000198f,\
  0.000053f,  -0.001473f, 0.001464f,  -0.003764f, 0.007051f,  -0.015617f, 0.046791f,  -0.007195f, -0.000401f, -0.000163f,\
  0.000193f,  -0.001532f, 0.001644f,  -0.004062f, 0.007825f,  -0.016871f, 0.046983f,  -0.006785f, -0.000467f, -0.000128f,\
  0.000331f,  -0.001578f, 0.001813f,  -0.004326f, 0.008539f,  -0.018006f, 0.047075f,  -0.006327f, -0.000542f, -0.000091f,\
  0.000467f,  -0.001609f, 0.001972f,  -0.004556f, 0.009193f,  -0.019022f, 0.047066f,  -0.005821f, -0.000627f, -0.000054f,\
  0.000599f,  -0.001625f, 0.002120f,  -0.004753f, 0.009787f,  -0.019919f, 0.046959f,  -0.005269f, -0.000721f, -0.000016f,\
  0.000728f,  -0.001625f, 0.002256f,  -0.004917f, 0.010319f,  -0.020697f, 0.046754f,  -0.004672f, -0.000824f, 0.000022f,
  0.000850f,  -0.001611f, 0.002381f,  -0.005047f, 0.010792f,  -0.021360f, 0.046453f,  -0.004029f, -0.000937f, 0.000061f,
  0.000968f,  -0.001581f, 0.002494f,  -0.005144f, 0.011206f,  -0.021908f, 0.046060f,  -0.003344f, -0.001057f, 0.000101f,
  0.001079f,  -0.001536f, 0.002596f,  -0.005209f, 0.011560f,  -0.022344f, 0.045575f,  -0.002618f, -0.001185f, 0.000141f,
  0.001183f,  -0.001476f, 0.002686f,  -0.005243f, 0.011857f,  -0.022671f, 0.045003f,  -0.001852f, -0.001321f, 0.000181f,
  0.001281f,  -0.001402f, 0.002765f,  -0.005246f, 0.012097f,  -0.022893f, 0.044345f,  -0.001048f, -0.001463f, 0.000221f,
  0.001373f,  -0.001315f, 0.002833f,  -0.005220f, 0.012283f,  -0.023011f, 0.043605f,  -0.000210f, -0.001611f, 0.000262f,
  0.001457f,  -0.001215f, 0.002889f,  -0.005165f, 0.012414f,  -0.023031f, 0.042787f,  0.000663f,  -0.001765f, 0.000302f,
  0.001535f,  -0.001102f, 0.002934f,  -0.005083f, 0.012494f,  -0.022954f, 0.041893f,  0.001566f,  -0.001923f, 0.000343f,
  0.001607f,  -0.000978f, 0.002969f,  -0.004975f, 0.012522f,  -0.022786f, 0.040928f,  0.002497f,  -0.002086f, 0.000383f,
  0.001672f,  -0.000844f, 0.002993f,  -0.004842f, 0.012503f,  -0.022531f, 0.039894f,  0.003456f,  -0.002252f, 0.000423f,
  0.001732f,  -0.000700f, 0.003006f,  -0.004685f, 0.012436f,  -0.022191f, 0.038796f,  0.004438f,  -0.002421f, 0.000463f,
  0.001785f,  -0.000547f, 0.003009f,  -0.004506f, 0.012324f,  -0.021772f, 0.037637f,  0.005441f,  -0.002591f, 0.000502f,
  0.001834f,  -0.000387f, 0.003002f,  -0.004307f, 0.012169f,  -0.021277f, 0.036421f,  0.006463f,  -0.002763f, 0.000540f,
  0.001877f,  -0.000220f, 0.002986f,  -0.004088f, 0.011973f,  -0.020712f, 0.035152f,  0.007501f,  -0.002935f, 0.000577f,
  0.001915f,  -0.000047f, 0.002960f,  -0.003851f, 0.011739f,  -0.020079f, 0.033834f,  0.008553f,  -0.003106f, 0.000614f,
  0.001949f,  0.000130f,  0.002925f,  -0.003598f, 0.011468f,  -0.019385f, 0.032470f,  0.009616f,  -0.003277f, 0.000649f,
  0.001978f,  0.000310f,  0.002881f,  -0.003331f, 0.011162f,  -0.018634f, 0.031066f,  0.010687f,  -0.003445f, 0.000683f,
  0.002004f,  0.000492f,  0.002829f,  -0.003051f, 0.010824f,  -0.017830f, 0.029625f,  0.011763f,  -0.003610f, 0.000716f,
  0.002027f,  0.000676f,  0.002768f,  -0.002759f, 0.010457f,  -0.016979f, 0.028151f,  0.012842f,  -0.003771f, 0.000747f,
  0.002046f,  0.000859f,  0.002700f,  -0.002458f, 0.010063f,  -0.016085f, 0.026649f,  0.013920f,  -0.003928f, 0.000776f,
  0.002062f,  0.001040f,  0.002626f,  -0.002150f, 0.009643f,  -0.015154f, 0.025124f,  0.014995f,  -0.004078f, 0.000804f,
  0.002076f,  0.001218f,  0.002544f,  -0.001836f, 0.009202f,  -0.014190f, 0.023579f,  0.016062f,  -0.004222f, 0.000829f,
  0.002087f,  0.001393f,  0.002456f,  -0.001518f, 0.008742f,  -0.013200f, 0.022020f,  0.017120f,  -0.004359f, 0.000853f,
  0.002096f,  0.001561f,  0.002363f,  -0.001199f, 0.008265f,  -0.012189f, 0.020451f,  0.018165f,  -0.004486f, 0.000874f,
  0.002103f,  0.001723f,  0.002266f,  -0.000881f, 0.007774f,  -0.011162f, 0.018877f,  0.019193f,  -0.004604f, 0.000892f,
  0.002108f,  0.001876f,  0.002164f,  -0.000566f, 0.007272f,  -0.010125f, 0.017303f,  0.020201f,  -0.004711f, 0.000908f,
  0.002111f,  0.002019f,  0.002059f,  -0.000256f, 0.006761f,  -0.009084f, 0.015735f,  0.021186f,  -0.004806f, 0.000921f,
  0.002111f,  0.002151f,  0.001951f,  0.000047f,  0.006246f,  -0.008045f, 0.014176f,  0.022145f,  -0.004887f, 0.000931f,
  0.002109f,  0.002271f,  0.001840f,  0.000340f,  0.005727f,  -0.007011f, 0.012631f,  0.023074f,  -0.004955f, 0.000937f,
  0.002105f,  0.002377f,  0.001729f,  0.000622f,  0.005208f,  -0.005989f, 0.011105f,  0.023970f,  -0.005008f, 0.000940f,
  0.002098f,  0.002469f,  0.001616f,  0.000891f,  0.004690f,  -0.004982f, 0.009601f,  0.024831f,  -0.005045f, 0.000940f,
  0.002087f,  0.002546f,  0.001503f,  0.001145f,  0.004176f,  -0.003995f, 0.008123f,  0.025654f,  -0.005065f, 0.000936f,
  0.002072f,  0.002607f,  0.001389f,  0.001384f,  0.003666f,  -0.003031f, 0.006673f,  0.026438f,  -0.005068f, 0.000928f,
  0.002053f,  0.002653f,  0.001275f,  0.001607f,  0.003162f,  -0.002091f, 0.005254f,  0.027181f,  -0.005054f, 0.000916f,
  0.002029f,  0.002682f,  0.001161f,  0.001813f,  0.002665f,  -0.001179f, 0.003868f,  0.027881f,  -0.005021f, 0.000900f,
  0.001999f,  0.002695f,  0.001048f,  0.002000f,  0.002176f,  -0.000297f, 0.002517f,  0.028538f,  -0.004970f, 0.000880f,
  0.001963f,  0.002693f,  0.000934f,  0.002170f,  0.001695f,  0.000554f,  0.001203f,  0.029151f,  -0.004900f, 0.000857f,
  0.001919f,  0.002674f,  0.000820f,  0.002321f,  0.001223f,  0.001373f,  -0.000075f, 0.029718f,  -0.004811f, 0.000829f,
  0.001869f,  0.002639f,  0.000706f,  0.002454f,  0.000759f,  0.002158f,  -0.001313f, 0.030239f,  -0.004703f, 0.000797f,
  0.001810f,  0.002589f,  0.000592f,  0.002568f,  0.000305f,  0.002909f,  -0.002512f, 0.030714f,  -0.004575f, 0.000761f,
  0.001743f,  0.002523f,  0.000478f,  0.002663f,  -0.000139f, 0.003624f,  -0.003670f, 0.031142f,  -0.004428f, 0.000720f,
  0.001666f,  0.002442f,  0.000364f,  0.002739f,  -0.000573f, 0.004303f,  -0.004787f, 0.031523f,  -0.004262f, 0.000676f,
  0.001580f,  0.002346f,  0.000249f,  0.002797f,  -0.000997f, 0.004945f,  -0.005862f, 0.031857f,  -0.004077f, 0.000628f,
  0.001485f,  0.002235f,  0.000133f,  0.002836f,  -0.001411f, 0.005550f,  -0.006895f, 0.032143f,  -0.003872f, 0.000575f,
  0.001379f,  0.002109f,  0.000017f,  0.002856f,  -0.001815f, 0.006117f,  -0.007885f, 0.032382f,  -0.003647f, 0.000518f,
  0.001263f,  0.001969f,  -0.000100f, 0.002858f,  -0.002208f, 0.006646f,  -0.008832f, 0.032573f,  -0.003404f, 0.000458f,
  0.001137f,  0.001815f,  -0.000217f, 0.002842f,  -0.002590f, 0.007137f,  -0.009736f, 0.032717f,  -0.003141f, 0.000393f,
  0.001001f,  0.001647f,  -0.000336f, 0.002808f,  -0.002962f, 0.007590f,  -0.010597f, 0.032814f,  -0.002860f, 0.000324f,
  0.000854f,  0.001466f,  -0.000456f, 0.002756f,  -0.003324f, 0.008004f,  -0.011414f, 0.032865f,  -0.002559f, 0.000252f,
  0.000698f,  0.001270f,  -0.000576f, 0.002687f,  -0.003674f, 0.008380f,  -0.012188f, 0.032869f,  -0.002241f, 0.000175f,
  0.000533f,  0.001061f,  -0.000698f, 0.002600f,  -0.004014f, 0.008718f,  -0.012918f, 0.032828f,  -0.001904f, 0.000095f,
  0.000359f,  0.000839f,  -0.000821f, 0.002497f,  -0.004344f, 0.009018f,  -0.013606f, 0.032741f,  -0.001550f, 0.000011f,
  0.000177f,  0.000604f,  -0.000944f, 0.002377f,  -0.004663f, 0.009281f,  -0.014251f, 0.032611f,  -0.001179f, -0.000076f,\
  -0.000012f, 0.000355f,  -0.001069f, 0.002242f,  -0.004971f, 0.009506f,  -0.014855f, 0.032437f,  -0.000791f, -0.000167f,\
  -0.000207f, 0.000092f,  -0.001195f, 0.002091f,  -0.005269f, 0.009695f,  -0.015417f, 0.032221f,  -0.000387f, -0.000261f,\
  -0.000407f, -0.000183f, -0.001321f, 0.001924f,  -0.005556f, 0.009848f,  -0.015939f, 0.031964f,  0.000033f,  -0.000358f,\
  -0.000611f, -0.000472f, -0.001448f, 0.001744f,  -0.005833f, 0.009966f,  -0.016421f, 0.031666f,  0.000468f,  -0.000458f,\
  -0.000818f, -0.000774f, -0.001576f, 0.001549f,  -0.006099f, 0.010050f,  -0.016864f, 0.031331f,  0.000917f,  -0.000561f,\
  -0.001026f, -0.001089f, -0.001704f, 0.001341f,  -0.006355f, 0.010101f,  -0.017269f, 0.030958f,  0.001379f,  -0.000666f,\
  -0.001235f, -0.001417f, -0.001833f, 0.001121f,  -0.006601f, 0.010119f,  -0.017637f, 0.030549f,  0.001854f,  -0.000775f,\
  -0.001443f, -0.001758f, -0.001962f, 0.000888f,  -0.006838f, 0.010106f,  -0.017970f, 0.030105f,  0.002341f,  -0.000885f,\
  -0.001650f, -0.002112f, -0.002091f, 0.000644f,  -0.007064f, 0.010063f,  -0.018267f, 0.029629f,  0.002840f,  -0.000998f,\
  -0.001854f, -0.002477f, -0.002220f, 0.000389f,  -0.007281f, 0.009990f,  -0.018532f, 0.029122f,  0.003349f,  -0.001112f,\
  -0.002054f, -0.002854f, -0.002350f, 0.000123f,  -0.007488f, 0.009890f,  -0.018764f, 0.028586f,  0.003867f,  -0.001229f,\
  -0.002251f, -0.003242f, -0.002480f, -0.000151f, -0.007686f, 0.009763f,  -0.018965f, 0.028021f,  0.004394f,  -0.001347f,\
  -0.002442f, -0.003640f, -0.002610f, -0.000434f, -0.007874f, 0.009610f,  -0.019137f, 0.027431f,  0.004929f,  -0.001466f,\
  -0.002628f, -0.004049f, -0.002741f, -0.000725f, -0.008054f, 0.009433f,  -0.019281f, 0.026817f,  0.005471f,  -0.001586f,\
  -0.002809f, -0.004465f, -0.002871f, -0.001023f, -0.008225f, 0.009234f,  -0.019398f, 0.026180f,  0.006020f,  -0.001708f,\
  -0.002983f, -0.004890f, -0.003003f, -0.001328f, -0.008388f, 0.009013f,  -0.019490f, 0.025523f,  0.006573f,  -0.001830f,\
  -0.003152f, -0.005322f, -0.003134f, -0.001637f, -0.008543f, 0.008772f,  -0.019558f, 0.024847f,  0.007131f,  -0.001953f,\
  -0.003314f, -0.005760f, -0.003266f, -0.001952f, -0.008689f, 0.008512f,  -0.019604f, 0.024155f,  0.007692f,  -0.002076f,\
  -0.003470f, -0.006202f, -0.003399f, -0.002270f, -0.008828f, 0.008235f,  -0.019629f, 0.023448f,  0.008255f,  -0.002199f,\
  -0.003620f, -0.006649f, -0.003533f, -0.002590f, -0.008960f, 0.007942f,  -0.019634f, 0.022728f,  0.008821f,  -0.002322f,\
  -0.003764f, -0.007098f, -0.003667f, -0.002913f, -0.009085f, 0.007635f,  -0.019622f, 0.021997f,  0.009387f,  -0.002445f,\
  -0.003903f, -0.007548f, -0.003803f, -0.003238f, -0.009203f, 0.007315f,  -0.019593f, 0.021257f,  0.009953f,  -0.002568f,\
  -0.004035f, -0.007999f, -0.003940f, -0.003563f, -0.009315f, 0.006983f,  -0.019549f, 0.020509f,  0.010519f,  -0.002690f,\
  -0.004163f, -0.008450f, -0.004078f, -0.003887f, -0.009421f, 0.006641f,  -0.019492f, 0.019755f,  0.011083f,  -0.002811f,\
  -0.004286f, -0.008899f, -0.004217f, -0.004211f, -0.009522f, 0.006290f,  -0.019421f, 0.018995f,  0.011644f,  -0.002931f,\
  -0.004405f, -0.009345f, -0.004358f, -0.004533f, -0.009617f, 0.005930f,  -0.019339f, 0.018233f,  0.012203f,  -0.003050f,\
  -0.004520f, -0.009787f, -0.004501f, -0.004853f, -0.009708f, 0.005565f,  -0.019247f, 0.017469f,  0.012759f,  -0.003168f,\
  -0.004632f, -0.010225f, -0.004645f, -0.005169f, -0.009794f, 0.005193f,  -0.019145f, 0.016704f,  0.013311f,  -0.003285f,\
  -0.004742f, -0.010656f, -0.004792f, -0.005482f, -0.009877f, 0.004817f,  -0.019035f, 0.015939f,  0.013858f,  -0.003400f,\
  -0.004850f, -0.011081f, -0.004941f, -0.005791f, -0.009956f, 0.004438f,  -0.018917f, 0.015177f,  0.014400f,  -0.003514f,\
  -0.004957f, -0.011499f, -0.005091f, -0.006095f, -0.010032f, 0.004057f,  -0.018793f, 0.014416f,  0.014937f,  -0.003625f,\
  -0.005064f, -0.011907f, -0.005244f, -0.006393f, -0.010106f, 0.003675f,  -0.018663f, 0.013659f,  0.015468f,  -0.003736f,\
  -0.005171f, -0.012306f, -0.005400f, -0.006686f, -0.010178f, 0.003292f,  -0.018528f, 0.012906f,  0.015992f,  -0.003844f,\
  -0.005280f, -0.012695f, -0.005558f, -0.006973f, -0.010249f, 0.002909f,  -0.018389f, 0.012158f,  0.016511f,  -0.003950f,\
  -0.005392f, -0.013073f, -0.005719f, -0.007252f, -0.010318f, 0.002528f,  -0.018246f, 0.011415f,  0.017022f,  -0.004055f,\
  -0.005506f, -0.013439f, -0.005882f, -0.007525f, -0.010387f, 0.002150f,  -0.018100f, 0.010679f,  0.017526f,  -0.004157f,\
  -0.005624f, -0.013792f, -0.006048f, -0.007790f, -0.010457f, 0.001774f,  -0.017951f, 0.009949f,  0.018024f,  -0.004257f,\
  -0.005746f, -0.014133f, -0.006217f, -0.008047f, -0.010526f, 0.001402f,  -0.017800f, 0.009226f,  0.018514f,  -0.004355f,\
  -0.005874f, -0.014460f, -0.006389f, -0.008296f, -0.010596f, 0.001034f,  -0.017647f, 0.008511f,  0.018996f,  -0.004451f,\
  -0.006008f, -0.014773f, -0.006564f, -0.008536f, -0.010668f, 0.000671f,  -0.017493f, 0.007804f,  0.019470f,  -0.004545f,\
  -0.006147f, -0.015071f, -0.006742f, -0.008768f, -0.010741f, 0.000313f,  -0.017338f, 0.007104f,  0.019937f,  -0.004637f,\
  -0.006294f, -0.015355f, -0.006923f, -0.008990f, -0.010816f, -0.000039f, -0.017183f, 0.006414f,  0.020396f,  -0.004726f,\
  -0.006448f, -0.015624f, -0.007107f, -0.009204f, -0.010893f, -0.000385f, -0.017027f, 0.005731f,  0.020846f,  -0.004813f,\
  -0.006608f, -0.015878f, -0.007294f, -0.009408f, -0.010972f, -0.000724f, -0.016870f, 0.005058f,  0.021289f,  -0.004898f,\
  -0.006777f, -0.016117f, -0.007483f, -0.009602f, -0.011054f, -0.001056f, -0.016714f, 0.004394f,  0.021723f,  -0.004981f,\
  -0.006952f, -0.016341f, -0.007676f, -0.009787f, -0.011139f, -0.001381f, -0.016558f, 0.003739f,  0.022149f,  -0.005061f,\
  -0.007135f, -0.016550f, -0.007872f, -0.009963f, -0.011226f, -0.001699f, -0.016403f, 0.003093f,  0.022567f,  -0.005139f,\
  -0.007324f, -0.016744f, -0.008070f, -0.010129f, -0.011316f, -0.002009f, -0.016248f, 0.002457f,  0.022976f,  -0.005214f,\
  -0.007520f, -0.016924f, -0.008271f, -0.010285f, -0.011408f, -0.002312f, -0.016094f, 0.001831f,  0.023377f,  -0.005287f,\
  -0.007722f, -0.017090f, -0.008474f, -0.010432f, -0.011503f, -0.002607f, -0.015941f, 0.001214f,  0.023770f,  -0.005358f,\
  -0.007929f, -0.017242f, -0.008680f, -0.010570f, -0.011600f, -0.002895f, -0.015788f, 0.000607f,  0.024154f,  -0.005427f,\
  -0.008141f, -0.017382f, -0.008888f, -0.010698f, -0.011699f, -0.003176f, -0.015636f, 0.000009f,  0.024529f,  -0.005493f,\
  -0.008357f, -0.017509f, -0.009098f, -0.010817f, -0.011801f, -0.003450f, -0.015485f, -0.000578f, 0.024896f,  -0.005556f,\
  -0.008575f, -0.017625f, -0.009309f, -0.010928f, -0.011903f, -0.003717f, -0.015335f, -0.001156f, 0.025255f,  -0.005618f,\
  -0.008797f, -0.017730f, -0.009522f, -0.011030f, -0.012007f, -0.003978f, -0.015186f, -0.001725f, 0.025605f,  -0.005677f,\
  -0.009019f, -0.017825f, -0.009736f, -0.011124f, -0.012112f, -0.004232f, -0.015038f, -0.002283f, 0.025947f,  -0.005733f,\
  -0.009243f, -0.017909f, -0.009950f, -0.011210f, -0.012217f, -0.004480f, -0.014890f, -0.002832f, 0.026280f,  -0.005787f,\
  -0.009466f, -0.017985f, -0.010165f, -0.011289f, -0.012322f, -0.004723f, -0.014743f, -0.003371f, 0.026605f,  -0.005839f,\
  -0.009689f, -0.018053f, -0.010380f, -0.011361f, -0.012426f, -0.004961f, -0.014596f, -0.003900f, 0.026921f,  -0.005889f,\
  -0.009911f, -0.018112f, -0.010595f, -0.011426f, -0.012529f, -0.005194f, -0.014451f, -0.004420f, 0.027230f,  -0.005935f,\
  -0.010132f, -0.018165f, -0.010810f, -0.011485f, -0.012631f, -0.005422f, -0.014305f, -0.004930f, 0.027529f,  -0.005980f,\
  -0.010351f, -0.018210f, -0.011023f, -0.011538f, -0.012731f, -0.005646f, -0.014160f, -0.005431f, 0.027821f,  -0.006022f,\
  -0.010568f, -0.018249f, -0.011234f, -0.011586f, -0.012829f, -0.005867f, -0.014015f, -0.005922f, 0.028104f,  -0.006062f,\
  -0.010783f, -0.018283f, -0.011444f, -0.011630f, -0.012923f, -0.006084f, -0.013870f, -0.006405f, 0.028380f,  -0.006100f,\
  -0.010996f, -0.018311f, -0.011651f, -0.011669f, -0.013015f, -0.006299f, -0.013726f, -0.006878f, 0.028647f,  -0.006135f,\
  -0.011206f, -0.018333f, -0.011856f, -0.011705f, -0.013102f, -0.006511f, -0.013581f, -0.007342f, 0.028906f,  -0.006167f,\
  -0.011415f, -0.018351f, -0.012057f, -0.011737f, -0.013185f, -0.006720f, -0.013436f, -0.007797f, 0.029157f,  -0.006198f,\
  -0.011621f, -0.018365f, -0.012254f, -0.011767f, -0.013263f, -0.006928f, -0.013291f, -0.008243f, 0.029400f,  -0.006226f,\
  -0.011825f, -0.018374f, -0.012447f, -0.011795f, -0.013336f, -0.007135f, -0.013145f, -0.008680f, 0.029636f,  -0.006251f,\
  -0.012027f, -0.018378f, -0.012636f, -0.011822f, -0.013404f, -0.007340f, -0.012999f, -0.009108f, 0.029863f,  -0.006275f,\
  -0.012228f, -0.018379f, -0.012820f, -0.011847f, -0.013465f, -0.007545f, -0.012853f, -0.009528f, 0.030083f,  -0.006296f,\
  -0.012427f, -0.018376f, -0.012998f, -0.011872f, -0.013521f, -0.007749f, -0.012706f, -0.009940f, 0.030295f,  -0.006314f,\
  -0.012626f, -0.018369f, -0.013171f, -0.011896f, -0.013569f, -0.007953f, -0.012558f, -0.010342f, 0.030500f,  -0.006331f,\
  -0.012824f, -0.018358f, -0.013337f, -0.011921f, -0.013611f, -0.008157f, -0.012410f, -0.010737f, 0.030697f,  -0.006345f,\
  -0.013021f, -0.018344f, -0.013498f, -0.011947f, -0.013645f, -0.008360f, -0.012260f, -0.011123f, 0.030887f,  -0.006356f,\
  -0.013218f, -0.018326f, -0.013651f, -0.011973f, -0.013672f, -0.008564f, -0.012110f, -0.011501f, 0.031069f,  -0.006366f,\
  -0.013414f, -0.018305f, -0.013798f, -0.012000f, -0.013692f, -0.008768f, -0.011960f, -0.011871f, 0.031244f,  -0.006373f,\
  -0.013611f, -0.018280f, -0.013938f, -0.012029f, -0.013704f, -0.008973f, -0.011808f, -0.012232f, 0.031411f,  -0.006377f,\
  -0.013808f, -0.018251f, -0.014071f, -0.012060f, -0.013708f, -0.009177f, -0.011655f, -0.012586f, 0.031571f,  -0.006380f,\
  -0.014005f, -0.018219f, -0.014197f, -0.012092f, -0.013705f, -0.009383f, -0.011502f, -0.012931f, 0.031724f,  -0.006380f,\
  -0.014203f, -0.018184f, -0.014315f, -0.012126f, -0.013693f, -0.009588f, -0.011348f, -0.013268f, 0.031869f,  -0.006378f,\
  -0.014401f, -0.018144f, -0.014426f, -0.012162f, -0.013675f, -0.009794f, -0.011193f, -0.013598f, 0.032008f,  -0.006373f,\
  -0.014600f, -0.018102f, -0.014529f, -0.012201f, -0.013648f, -0.010000f, -0.011038f, -0.013919f, 0.032139f,  -0.006367f,\
  -0.014799f, -0.018055f, -0.014626f, -0.012241f, -0.013614f, -0.010205f, -0.010882f, -0.014233f, 0.032262f,  -0.006358f,\
  -0.014998f, -0.018005f, -0.014715f, -0.012283f, -0.013572f, -0.010411f, -0.010725f, -0.014538f, 0.032379f,  -0.006346f,\
  -0.015198f, -0.017951f, -0.014796f, -0.012327f, -0.013524f, -0.010616f, -0.010568f, -0.014836f, 0.032488f,  -0.006333f,\
  -0.015397f, -0.017893f, -0.014871f, -0.012373f, -0.013468f, -0.010821f, -0.010410f, -0.015126f, 0.032591f,  -0.006317f,\
  -0.015597f, -0.017832f, -0.014939f, -0.012420f, -0.013405f, -0.011025f, -0.010252f, -0.015407f, 0.032686f,  -0.006299f,\
  -0.015797f, -0.017766f, -0.015000f, -0.012470f, -0.013336f, -0.011228f, -0.010094f, -0.015681f, 0.032774f,  -0.006278f,\
  -0.015997f, -0.017697f, -0.015054f, -0.012521f, -0.013260f, -0.011430f, -0.009936f, -0.015947f, 0.032854f,  -0.006256f,\
  -0.016197f, -0.017623f, -0.015101f, -0.012573f, -0.013178f, -0.011630f, -0.009778f, -0.016205f, 0.032928f,  -0.006231f,\
  -0.016396f, -0.017545f, -0.015142f, -0.012627f, -0.013090f, -0.011829f, -0.009620f, -0.016456f, 0.032994f,  -0.006204f,\
  -0.016595f, -0.017464f, -0.015177f, -0.012681f, -0.012997f, -0.012025f, -0.009461f, -0.016698f, 0.033053f,  -0.006174f,\
  -0.016793f, -0.017378f, -0.015206f, -0.012737f, -0.012897f, -0.012220f, -0.009304f, -0.016932f, 0.033105f,  -0.006142f,\
  -0.016991f, -0.017287f, -0.015228f, -0.012793f, -0.012793f, -0.012412f, -0.009146f, -0.017159f, 0.033150f,  -0.006108f,\
  -0.017188f, -0.017193f, -0.015245f, -0.012851f, -0.012683f, -0.012601f, -0.008989f, -0.017377f, 0.033187f,  -0.006072f,\
  -0.017383f, -0.017094f, -0.015256f, -0.012908f, -0.012568f, -0.012788f, -0.008832f, -0.017588f, 0.033218f,  -0.006033f,\
  -0.017577f, -0.016990f, -0.015261f, -0.012966f, -0.012449f, -0.012971f, -0.008676f, -0.017791f, 0.033241f,  -0.005993f,\
  -0.017770f, -0.016882f, -0.015261f, -0.013025f, -0.012326f, -0.013151f, -0.008521f, -0.017985f, 0.033257f,  -0.005949f,\
  -0.017961f, -0.016769f, -0.015256f, -0.013083f, -0.012198f, -0.013327f, -0.008366f, -0.018172f, 0.033265f,  -0.005904f,\
  -0.018151f, -0.016652f, -0.015246f, -0.013141f, -0.012067f, -0.013500f, -0.008213f, -0.018351f, 0.033267f,  -0.005856f,\
  -0.018339f, -0.016530f, -0.015231f, -0.013199f, -0.011931f, -0.013668f, -0.008060f, -0.018521f, 0.033261f,  -0.005806f,\
  -0.018525f, -0.016403f, -0.015211f, -0.013257f, -0.011793f, -0.013832f, -0.007909f, -0.018684f, 0.033248f,  -0.005754f,\
  -0.018708f, -0.016272f, -0.015186f, -0.013314f, -0.011651f, -0.013991f, -0.007758f, -0.018839f, 0.033227f,  -0.005700f,\
  -0.018889f, -0.016136f, -0.015158f, -0.013370f, -0.011506f, -0.014146f, -0.007609f, -0.018985f, 0.033200f,  -0.005643f,\
  -0.019068f, -0.015995f, -0.015124f, -0.013425f, -0.011358f, -0.014296f, -0.007461f, -0.019124f, 0.033165f,  -0.005584f,\
  -0.019244f, -0.015849f, -0.015087f, -0.013479f, -0.011208f, -0.014440f, -0.007315f, -0.019255f, 0.033123f,  -0.005522f,\
  -0.019417f, -0.015698f, -0.015046f, -0.013532f, -0.011055f, -0.014579f, -0.007169f, -0.019377f, 0.033073f,  -0.005459f,\
  -0.019588f, -0.015543f, -0.015001f, -0.013583f, -0.010900f, -0.014713f, -0.007026f, -0.019492f, 0.033017f,  -0.005393f,\
  -0.019755f, -0.015382f, -0.014952f, -0.013633f, -0.010743f, -0.014841f, -0.006884f, -0.019599f, 0.032953f,  -0.005324f,\
  -0.019919f, -0.015217f, -0.014900f, -0.013681f, -0.010585f, -0.014963f, -0.006743f, -0.019697f, 0.032881f,  -0.005254f,\
  -0.020080f, -0.015047f, -0.014844f, -0.013727f, -0.010424f, -0.015079f, -0.006605f, -0.019788f, 0.032803f,  -0.005181f,\
  -0.020237f, -0.014872f, -0.014785f, -0.013771f, -0.010263f, -0.015188f, -0.006468f, -0.019870f, 0.032717f,  -0.005106f,\
  -0.020391f, -0.014692f, -0.014722f, -0.013814f, -0.010100f, -0.015292f, -0.006332f, -0.019945f, 0.032624f,  -0.005029f,\
  -0.020541f, -0.014507f, -0.014657f, -0.013853f, -0.009935f, -0.015389f, -0.006199f, -0.020012f, 0.032524f,  -0.004949f,\
  -0.020687f, -0.014318f, -0.014589f, -0.013891f, -0.009770f, -0.015479f, -0.006068f, -0.020070f, 0.032416f,  -0.004867f,\
  -0.020829f, -0.014123f, -0.014517f, -0.013926f, -0.009604f, -0.015563f, -0.005938f, -0.020121f, 0.032301f,  -0.004783f,\
  -0.020967f, -0.013924f, -0.014444f, -0.013958f, -0.009438f, -0.015640f, -0.005810f, -0.020164f, 0.032179f,  -0.004696f,\
  -0.021101f, -0.013721f, -0.014367f, -0.013987f, -0.009271f, -0.015710f, -0.005685f, -0.020199f, 0.032050f,  -0.004608f,\
  -0.021231f, -0.013513f, -0.014288f, -0.014013f, -0.009103f, -0.015774f, -0.005561f, -0.020226f, 0.031914f,  -0.004517f,\
  -0.021356f, -0.013300f, -0.014207f, -0.014037f, -0.008936f, -0.015830f, -0.005440f, -0.020245f, 0.031771f,  -0.004424f,\
  -0.021477f, -0.013083f, -0.014123f, -0.014057f, -0.008768f, -0.015880f, -0.005320f, -0.020256f, 0.031620f,  -0.004328f,\
  -0.021593f, -0.012862f, -0.014037f, -0.014073f, -0.008600f, -0.015922f, -0.005202f, -0.020260f, 0.031463f,  -0.004231f,\
  -0.021704f, -0.012637f, -0.013949f, -0.014087f, -0.008432f, -0.015958f, -0.005087f, -0.020256f, 0.031298f,  -0.004131f,\
  -0.021811f, -0.012407f, -0.013859f, -0.014097f, -0.008265f, -0.015986f, -0.004973f, -0.020245f, 0.031127f,  -0.004029f,\
  -0.021913f, -0.012174f, -0.013767f, -0.014103f, -0.008097f, -0.016008f, -0.004861f, -0.020225f, 0.030949f,  -0.003925f,\
  -0.022010f, -0.011937f, -0.013673f, -0.014105f, -0.007930f, -0.016023f, -0.004752f, -0.020199f, 0.030764f,  -0.003819f,\
  -0.022102f, -0.011697f, -0.013577f, -0.014104f, -0.007764f, -0.016031f, -0.004644f, -0.020165f, 0.030572f,  -0.003710f,\
  -0.022189f, -0.011453f, -0.013480f, -0.014099f, -0.007597f, -0.016032f, -0.004538f, -0.020124f, 0.030374f,  -0.003600f,\
  -0.022272f, -0.011205f, -0.013381f, -0.014090f, -0.007432f, -0.016027f, -0.004434f, -0.020075f, 0.030169f,  -0.003487f,\
  -0.022349f, -0.010955f, -0.013280f, -0.014077f, -0.007267f, -0.016015f, -0.004332f, -0.020019f, 0.029957f,  -0.003372f,\
  -0.022421f, -0.010702f, -0.013177f, -0.014060f, -0.007102f, -0.015996f, -0.004232f, -0.019956f, 0.029739f,  -0.003256f,\
  -0.022487f, -0.010446f, -0.013073f, -0.014038f, -0.006939f, -0.015971f, -0.004133f, -0.019886f, 0.029514f,  -0.003137f,\
  -0.022549f, -0.010187f, -0.012967f, -0.014013f, -0.006776f, -0.015940f, -0.004036f, -0.019809f, 0.029283f,  -0.003016f,\
  -0.022605f, -0.009926f, -0.012860f, -0.013983f, -0.006614f, -0.015902f, -0.003941f, -0.019725f, 0.029046f,  -0.002893f,\
  -0.022656f, -0.009663f, -0.012752f, -0.013948f, -0.006452f, -0.015859f, -0.003848f, -0.019635f, 0.028803f,  -0.002768f,\
  -0.022701f, -0.009398f, -0.012642f, -0.013910f, -0.006292f, -0.015809f, -0.003756f, -0.019537f, 0.028553f,  -0.002641f,\
  -0.022741f, -0.009131f, -0.012531f, -0.013866f, -0.006133f, -0.015752f, -0.003666f, -0.019433f, 0.028298f,  -0.002513f,\
  -0.022775f, -0.008862f, -0.012419f, -0.013818f, -0.005974f, -0.015690f, -0.003578f, -0.019322f, 0.028036f,  -0.002382f,\
  -0.022804f, -0.008591f, -0.012305f, -0.013766f, -0.005817f, -0.015622f, -0.003491f, -0.019205f, 0.027769f,  -0.002249f,\
  -0.022827f, -0.008319f, -0.012191f, -0.013708f, -0.005661f, -0.015549f, -0.003406f, -0.019081f, 0.027495f,  -0.002115f,\
  -0.022845f, -0.008046f, -0.012075f, -0.013646f, -0.005506f, -0.015469f, -0.003322f, -0.018951f, 0.027216f,  -0.001979f,\
  -0.022856f, -0.007771f, -0.011959f, -0.013579f, -0.005353f, -0.015384f, -0.003240f, -0.018815f, 0.026931f,  -0.001841f,\
  -0.022862f, -0.007496f, -0.011841f, -0.013508f, -0.005200f, -0.015293f, -0.003159f, -0.018673f, 0.026641f,  -0.001701f,\
  -0.022862f, -0.007220f, -0.011723f, -0.013431f, -0.005049f, -0.015197f, -0.003080f, -0.018524f, 0.026345f,  -0.001559f,\
  -0.022856f, -0.006943f, -0.011603f, -0.013349f, -0.004899f, -0.015096f, -0.003002f, -0.018370f, 0.026044f,  -0.001416f,\
  -0.022845f, -0.006666f, -0.011483f, -0.013263f, -0.004751f, -0.014989f, -0.002925f, -0.018210f, 0.025737f,  -0.001271f,\
  -0.022827f, -0.006389f, -0.011362f, -0.013172f, -0.004604f, -0.014877f, -0.002850f, -0.018043f, 0.025425f,  -0.001124f,\
  -0.022803f, -0.006111f, -0.011240f, -0.013075f, -0.004458f, -0.014760f, -0.002776f, -0.017872f, 0.025108f,  -0.000976f,\
  -0.022774f, -0.005834f, -0.011117f, -0.012974f, -0.004314f, -0.014638f, -0.002703f, -0.017694f, 0.024786f,  -0.000826f,\
  -0.022738f, -0.005556f, -0.010994f, -0.012868f, -0.004172f, -0.014511f, -0.002632f, -0.017511f, 0.024459f,  -0.000675f,\
  -0.022696f, -0.005279f, -0.010870f, -0.012756f, -0.004031f, -0.014379f, -0.002562f, -0.017323f, 0.024127f,  -0.000522f,\
  -0.022648f, -0.005003f, -0.010745f, -0.012640f, -0.003891f, -0.014243f, -0.002493f, -0.017129f, 0.023790f,  -0.000368f,\
  -0.022594f, -0.004727f, -0.010620f, -0.012519f, -0.003754f, -0.014102f, -0.002425f, -0.016930f, 0.023449f,  -0.000212f,\
  -0.022533f, -0.004452f, -0.010494f, -0.012393f, -0.003617f, -0.013956f, -0.002358f, -0.016726f, 0.023103f,  -0.000055f,\
  -0.022467f, -0.004178f, -0.010368f, -0.012262f, -0.003483f, -0.013806f, -0.002293f, -0.016517f, 0.022753f,  0.000104f,\
  -0.022394f, -0.003905f, -0.010241f, -0.012126f, -0.003350f, -0.013652f, -0.002228f, -0.016303f, 0.022398f,  0.000264f,\
  -0.022315f, -0.003633f, -0.010114f, -0.011985f, -0.003219f, -0.013493f, -0.002165f, -0.016085f, 0.022039f,  0.000425f,\
  -0.022230f, -0.003362f, -0.009986f, -0.011839f, -0.003090f, -0.013331f, -0.002102f, -0.015861f, 0.021677f,  0.000587f,\
  -0.022138f, -0.003093f, -0.009858f, -0.011689f, -0.002963f, -0.013164f, -0.002041f, -0.015634f, 0.021310f,  0.000750f,\
  -0.022040f, -0.002826f, -0.009730f, -0.011534f, -0.002838f, -0.012993f, -0.001981f, -0.015402f, 0.020939f,  0.000914f,\
  -0.021936f, -0.002560f, -0.009601f, -0.011374f, -0.002714f, -0.012819f, -0.001922f, -0.015165f, 0.020565f,  0.001080f,\
  -0.021826f, -0.002297f, -0.009472f, -0.011210f, -0.002592f, -0.012641f, -0.001864f, -0.014925f, 0.020188f,  0.001246f,\
  -0.021709f, -0.002035f, -0.009342f, -0.011042f, -0.002472f, -0.012459f, -0.001807f, -0.014681f, 0.019807f,  0.001413f,\
  -0.021586f, -0.001775f, -0.009213f, -0.010869f, -0.002355f, -0.012274f, -0.001750f, -0.014433f, 0.019423f,  0.001581f,\
  -0.021457f, -0.001518f, -0.009083f, -0.010693f, -0.002239f, -0.012086f, -0.001695f, -0.014181f, 0.019037f,  0.001750f,\
  -0.021322f, -0.001263f, -0.008952f, -0.010512f, -0.002125f, -0.011894f, -0.001641f, -0.013927f, 0.018647f,  0.001919f,\
  -0.021180f, -0.001011f, -0.008822f, -0.010327f, -0.002013f, -0.011699f, -0.001587f, -0.013668f, 0.018255f,  0.002088f,\
  -0.021033f, -0.000762f, -0.008691f, -0.010139f, -0.001903f, -0.011501f, -0.001535f, -0.013407f, 0.017860f,  0.002259f,\
  -0.020879f, -0.000515f, -0.008560f, -0.009947f, -0.001795f, -0.011301f, -0.001484f, -0.013143f, 0.017464f,  0.002429f,\
  -0.020719f, -0.000271f, -0.008429f, -0.009751f, -0.001689f, -0.011097f, -0.001433f, -0.012876f, 0.017065f,  0.002600f,\
  -0.020554f, -0.000030f, -0.008298f, -0.009552f, -0.001585f, -0.010891f, -0.001383f, -0.012606f, 0.016665f,  0.002771f,\
  -0.020382f, 0.000208f,  -0.008166f, -0.009351f, -0.001484f, -0.010683f, -0.001334f, -0.012334f, 0.016262f,  0.002942f,\
  -0.020204f, 0.000443f,  -0.008035f, -0.009146f, -0.001384f, -0.010472f, -0.001286f, -0.012060f, 0.015859f,  0.003114f,\
  -0.020020f, 0.000674f,  -0.007903f, -0.008938f, -0.001287f, -0.010259f, -0.001239f, -0.011784f, 0.015454f,  0.003285f,\
  -0.019831f, 0.000902f,  -0.007771f, -0.008728f, -0.001191f, -0.010044f, -0.001193f, -0.011506f, 0.015048f,  0.003456f,\
  -0.019635f, 0.001126f,  -0.007639f, -0.008515f, -0.001098f, -0.009826f, -0.001148f, -0.011227f, 0.014642f,  0.003627f,\
  -0.019434f, 0.001347f,  -0.007507f, -0.008300f, -0.001007f, -0.009607f, -0.001103f, -0.010945f, 0.014234f,  0.003797f,\
  -0.019227f, 0.001564f,  -0.007374f, -0.008083f, -0.000918f, -0.009386f, -0.001059f, -0.010663f, 0.013826f,  0.003967f,\
  -0.019015f, 0.001777f,  -0.007242f, -0.007864f, -0.000832f, -0.009164f, -0.001017f, -0.010379f, 0.013418f,  0.004137f,\
  -0.018796f, 0.001986f,  -0.007110f, -0.007643f, -0.000747f, -0.008939f, -0.000975f, -0.010095f, 0.013010f,  0.004306f,\
  -0.018572f, 0.002190f,  -0.006977f, -0.007421f, -0.000665f, -0.008714f, -0.000933f, -0.009809f, 0.012602f,  0.004474f,\
  -0.018343f, 0.002391f,  -0.006844f, -0.007197f, -0.000585f, -0.008487f, -0.000893f, -0.009523f, 0.012195f,  0.004641f,\
  -0.018108f, 0.002587f,  -0.006712f, -0.006972f, -0.000507f, -0.008259f, -0.000853f, -0.009237f, 0.011788f,  0.004808f,\
  -0.017867f, 0.002779f,  -0.006579f, -0.006745f, -0.000432f, -0.008030f, -0.000815f, -0.008950f, 0.011381f,  0.004974f,\
  -0.017621f, 0.002967f,  -0.006446f, -0.006518f, -0.000359f, -0.007800f, -0.000777f, -0.008663f, 0.010976f,  0.005139f,\
  -0.017370f, 0.003150f,  -0.006313f, -0.006290f, -0.000288f, -0.007569f, -0.000740f, -0.008376f, 0.010571f,  0.005302f,\
  -0.017114f, 0.003328f,  -0.006180f, -0.006061f, -0.000219f, -0.007337f, -0.000703f, -0.008090f, 0.010168f,  0.005465f,\
  -0.016852f, 0.003501f,  -0.006048f, -0.005832f, -0.000153f, -0.007105f, -0.000668f, -0.007804f, 0.009767f,  0.005626f,\
  -0.016585f, 0.003670f,  -0.005915f, -0.005602f, -0.000089f, -0.006872f, -0.000633f, -0.007518f, 0.009366f,  0.005786f,\
  -0.016313f, 0.003834f,  -0.005782f, -0.005372f, -0.000027f, -0.006639f, -0.000599f, -0.007233f, 0.008968f,  0.005945f,\
  -0.016036f, 0.003992f,  -0.005649f, -0.005142f, 0.000033f,  -0.006406f, -0.000566f, -0.006948f, 0.008572f,  0.006102f,\
  -0.015754f, 0.004146f,  -0.005516f, -0.004912f, 0.000090f,  -0.006172f, -0.000533f, -0.006665f, 0.008177f,  0.006257f,\
  -0.015467f, 0.004294f,  -0.005383f, -0.004683f, 0.000145f,  -0.005938f, -0.000501f, -0.006383f, 0.007785f,  0.006411f,\
  -0.015175f, 0.004437f,  -0.005250f, -0.004453f, 0.000197f,  -0.005705f, -0.000470f, -0.006102f, 0.007395f,  0.006563f,\
  -0.014878f, 0.004575f,  -0.005117f, -0.004225f, 0.000248f,  -0.005471f, -0.000440f, -0.005823f, 0.007008f,  0.006714f,\
  -0.014577f, 0.004707f,  -0.004984f, -0.003997f, 0.000296f,  -0.005238f, -0.000410f, -0.005545f, 0.006624f,  0.006862f,\
  -0.014271f, 0.004833f,  -0.004851f, -0.003770f, 0.000341f,  -0.005005f, -0.000382f, -0.005269f, 0.006243f,  0.007009f,\
  -0.013960f, 0.004954f,  -0.004718f, -0.003543f, 0.000385f,  -0.004772f, -0.000354f, -0.004994f, 0.005864f,  0.007154f,\
  -0.013644f, 0.005070f,  -0.004585f, -0.003318f, 0.000426f,  -0.004541f, -0.000326f, -0.004722f, 0.005489f,  0.007296f,\
  -0.013324f, 0.005179f,  -0.004452f, -0.003094f, 0.000465f,  -0.004309f, -0.000299f, -0.004451f, 0.005117f,  0.007437f,\
  -0.013000f, 0.005283f,  -0.004320f, -0.002872f, 0.000501f,  -0.004079f, -0.000273f, -0.004183f, 0.004749f,  0.007575f,\
  -0.012671f, 0.005381f,  -0.004187f, -0.002651f, 0.000535f,  -0.003849f, -0.000248f, -0.003917f, 0.004384f,  0.007712f,\
  -0.012337f, 0.005472f,  -0.004054f, -0.002431f, 0.000567f,  -0.003620f, -0.000223f, -0.003654f, 0.004023f,  0.007846f,\
  -0.012000f, 0.005558f,  -0.003921f, -0.002214f, 0.000597f,  -0.003393f, -0.000199f, -0.003393f, 0.003666f,  0.007977f,\
  -0.011658f, 0.005638f,  -0.003788f, -0.001998f, 0.000625f,  -0.003166f, -0.000176f, -0.003135f, 0.003313f,  0.008106f,\
  -0.011312f, 0.005712f,  -0.003655f, -0.001784f, 0.000650f,  -0.002941f, -0.000153f, -0.002880f, 0.002964f,  0.008233f,\
  -0.010962f, 0.005779f,  -0.003522f, -0.001572f, 0.000673f,  -0.002717f, -0.000131f, -0.002627f, 0.002619f,  0.008358f,\
  -0.010608f, 0.005840f,  -0.003389f, -0.001362f, 0.000694f,  -0.002495f, -0.000110f, -0.002378f, 0.002278f,  0.008480f,\
  -0.010250f, 0.005895f,  -0.003256f, -0.001155f, 0.000713f,  -0.002274f, -0.000089f, -0.002131f, 0.001942f,  0.008599f,\
  -0.009889f, 0.005943f,  -0.003123f, -0.000950f, 0.000730f,  -0.002054f, -0.000069f, -0.001888f, 0.001610f,  0.008716f,\
  -0.009523f, 0.005985f,  -0.002990f, -0.000747f, 0.000745f,  -0.001837f, -0.000049f, -0.001648f, 0.001283f,  0.008830f,\
  -0.009154f, 0.006021f,  -0.002857f, -0.000547f, 0.000758f,  -0.001621f, -0.000030f, -0.001411f, 0.000961f,  0.008941f,\
  -0.008782f, 0.006051f,  -0.002724f, -0.000350f, 0.000769f,  -0.001407f, -0.000011f, -0.001178f, 0.000643f,  0.009050f,\
  -0.008406f, 0.006073f,  -0.002590f, -0.000155f, 0.000778f,  -0.001195f, 0.000007f,  -0.000948f, 0.000330f,  0.009156f,\
  -0.008027f, 0.006090f,  -0.002457f, 0.000036f,  0.000785f,  -0.000985f, 0.000024f,  -0.000722f, 0.000022f,  0.009260f,\
  -0.007644f, 0.006100f,  -0.002323f, 0.000225f,  0.000791f,  -0.000778f, 0.000041f,  -0.000499f, -0.000281f, 0.009361f,\
  -0.007258f, 0.006103f,  -0.002190f, 0.000411f,  0.000795f,  -0.000572f, 0.000058f,  -0.000280f, -0.000580f, 0.009459f,\
  -0.006869f, 0.006100f,  -0.002056f, 0.000593f,  0.000797f,  -0.000369f, 0.000074f,  -0.000064f, -0.000873f, 0.009554f,\
  -0.006478f, 0.006091f,  -0.001922f, 0.000773f,  0.000798f,  -0.000168f, 0.000090f,  0.000148f,  -0.001161f, 0.009647f,\
  -0.006083f, 0.006075f,  -0.001788f, 0.000949f,  0.000797f,  0.000030f,  0.000105f,  0.000356f,  -0.001444f, 0.009737f,\
  -0.005686f, 0.006053f,  -0.001653f, 0.001122f,  0.000795f,  0.000225f,  0.000120f,  0.000561f,  -0.001723f, 0.009824f,\
  -0.005287f, 0.006024f,  -0.001519f, 0.001291f,  0.000792f,  0.000418f,  0.000135f,  0.000761f,  -0.001996f, 0.009908f,\
  -0.004884f, 0.005989f,  -0.001384f, 0.001457f,  0.000787f,  0.000609f,  0.000149f,  0.000958f,  -0.002264f, 0.009990f,\
  -0.004480f, 0.005947f,  -0.001249f, 0.001620f,  0.000781f,  0.000796f,  0.000163f,  0.001152f,  -0.002527f, 0.010069f,\
  -0.004074f, 0.005899f,  -0.001113f, 0.001779f,  0.000775f,  0.000980f,  0.000177f,  0.001341f,  -0.002784f, 0.010145f,\
  -0.003665f, 0.005845f,  -0.000978f, 0.001935f,  0.000767f,  0.001162f,  0.000191f,  0.001526f,  -0.003037f, 0.010219f,\
  -0.003255f, 0.005784f,  -0.000842f, 0.002087f,  0.000759f,  0.001340f,  0.000204f,  0.001708f,  -0.003285f, 0.010290f,\
  -0.002842f, 0.005718f,  -0.000705f, 0.002235f,  0.000749f,  0.001515f,  0.000217f,  0.001886f,  -0.003527f, 0.010358f,\
  -0.002429f, 0.005645f,  -0.000569f, 0.002380f,  0.000739f,  0.001688f,  0.000230f,  0.002060f,  -0.003765f, 0.010424f,\
  -0.002013f, 0.005566f,  -0.000432f, 0.002521f,  0.000728f,  0.001856f,  0.000243f,  0.002230f,  -0.003997f, 0.010487f,\
  -0.001597f, 0.005481f,  -0.000294f, 0.002658f,  0.000717f,  0.002022f,  0.000255f,  0.002397f,  -0.004224f, 0.010548f,\
  -0.001179f, 0.005390f,  -0.000156f, 0.002791f,  0.000706f,  0.002184f,  0.000268f,  0.002559f,  -0.004446f, 0.010606f,\
  -0.000760f, 0.005293f,  -0.000018f, 0.002921f,  0.000694f,  0.002343f,  0.000280f,  0.002718f,  -0.004664f, 0.010661f,\
  -0.000340f, 0.005190f,  0.000120f,  0.003047f,  0.000681f,  0.002498f,  0.000292f,  0.002873f,  -0.004876f, 0.010714f,\
  0.000080f,  0.005081f,  0.000259f,  0.003168f,  0.000669f,  0.002649f,  0.000305f,  0.003024f,  -0.005083f, 0.010764f,\
  0.000501f,  0.004966f,  0.000399f,  0.003286f,  0.000656f,  0.002797f,  0.000317f,  0.003171f,  -0.005285f, 0.010812f,\
  0.000923f,  0.004846f,  0.000539f,  0.003400f,  0.000643f,  0.002942f,  0.000329f,  0.003314f,  -0.005482f, 0.010858f,\
  0.001345f,  0.004720f,  0.000679f,  0.003510f,  0.000631f,  0.003082f,  0.000341f,  0.003454f,  -0.005674f, 0.010900f,\
  0.001767f,  0.004588f,  0.000819f,  0.003616f,  0.000618f,  0.003219f,  0.000354f,  0.003589f,  -0.005861f, 0.010941f,\
  0.002189f,  0.004452f,  0.000960f,  0.003718f,  0.000605f,  0.003352f,  0.000366f,  0.003721f,  -0.006043f, 0.010979f,\
  0.002611f,  0.004309f,  0.001102f,  0.003816f,  0.000593f,  0.003482f,  0.000378f,  0.003849f,  -0.006220f, 0.011015f,\
  0.003032f,  0.004162f,  0.001244f,  0.003910f,  0.000581f,  0.003607f,  0.000391f,  0.003974f,  -0.006392f, 0.011048f,\
  0.003453f,  0.004009f,  0.001386f,  0.004000f,  0.000569f,  0.003728f,  0.000403f,  0.004094f,  -0.006559f, 0.011079f,\
  0.003874f,  0.003851f,  0.001528f,  0.004085f,  0.000558f,  0.003846f,  0.000416f,  0.004211f,  -0.006722f, 0.011107f,\
  0.004293f,  0.003688f,  0.001671f,  0.004167f,  0.000547f,  0.003959f,  0.000429f,  0.004324f,  -0.006879f, 0.011133f,\
  0.004712f,  0.003521f,  0.001814f,  0.004245f,  0.000537f,  0.004069f,  0.000442f,  0.004433f,  -0.007032f, 0.011157f,\
  0.005129f,  0.003349f,  0.001957f,  0.004319f,  0.000528f,  0.004174f,  0.000455f,  0.004539f,  -0.007180f, 0.011179f,\
  0.005546f,  0.003172f,  0.002101f,  0.004388f,  0.000518f,  0.004275f,  0.000468f,  0.004641f,  -0.007323f, 0.011198f,\
  0.005960f,  0.002991f,  0.002245f,  0.004454f,  0.000510f,  0.004373f,  0.000481f,  0.004739f,  -0.007462f, 0.011216f,\
  0.006374f,  0.002805f,  0.002389f,  0.004516f,  0.000502f,  0.004466f,  0.000495f,  0.004834f,  -0.007596f, 0.011231f,\
  0.006785f,  0.002616f,  0.002533f,  0.004574f,  0.000495f,  0.004555f,  0.000508f,  0.004925f,  -0.007725f, 0.011244f,\
  0.007194f,  0.002422f,  0.002677f,  0.004628f,  0.000489f,  0.004641f,  0.000522f,  0.005013f,  -0.007850f, 0.011254f,\
  0.007601f,  0.002225f,  0.002821f,  0.004678f,  0.000483f,  0.004722f,  0.000536f,  0.005097f,  -0.007970f, 0.011263f,\
  0.008006f,  0.002024f,  0.002965f,  0.004724f,  0.000478f,  0.004800f,  0.000550f,  0.005177f,  -0.008085f, 0.011269f,\
  0.008409f,  0.001819f,  0.003109f,  0.004767f,  0.000474f,  0.004873f,  0.000564f,  0.005255f,  -0.008196f, 0.011274f,\
  0.008808f,  0.001612f,  0.003253f,  0.004806f,  0.000471f,  0.004943f,  0.000579f,  0.005328f,  -0.008303f, 0.011276f,\
  0.009205f,  0.001401f,  0.003396f,  0.004841f,  0.000468f,  0.005009f,  0.000593f,  0.005399f,  -0.008405f, 0.011277f,\
  0.009599f,  0.001188f,  0.003539f,  0.004873f,  0.000467f,  0.005071f,  0.000608f,  0.005466f,  -0.008503f, 0.011275f,\
  0.009989f,  0.000972f,  0.003682f,  0.004901f,  0.000466f,  0.005129f,  0.000622f,  0.005530f,  -0.008597f, 0.011272f,\
  0.010376f,  0.000754f,  0.003825f,  0.004926f,  0.000465f,  0.005184f,  0.000637f,  0.005591f,  -0.008687f, 0.011267f,\
  0.010759f,  0.000534f,  0.003967f,  0.004947f,  0.000466f,  0.005235f,  0.000652f,  0.005649f,  -0.008773f, 0.011260f,\
  0.011139f,  0.000311f,  0.004108f,  0.004965f,  0.000467f,  0.005282f,  0.000667f,  0.005703f,  -0.008854f, 0.011251f,\
  0.011514f,  0.000087f,  0.004248f,  0.004980f,  0.000469f,  0.005326f,  0.000682f,  0.005755f,  -0.008932f, 0.011240f,\
  0.011886f,  -0.000138f, 0.004388f,  0.004992f,  0.000472f,  0.005367f,  0.000698f,  0.005803f,  -0.009005f, 0.011228f,\
  0.012253f,  -0.000365f, 0.004527f,  0.005001f,  0.000476f,  0.005404f,  0.000713f,  0.005849f,  -0.009075f, 0.011214f,\
  0.012615f,  -0.000592f, 0.004665f,  0.005007f,  0.000480f,  0.005439f,  0.000728f,  0.005892f,  -0.009142f, 0.011198f,\
  0.012973f,  -0.000820f, 0.004802f,  0.005011f,  0.000485f,  0.005470f,  0.000744f,  0.005933f,  -0.009204f, 0.011181f,\
  0.013326f,  -0.001048f, 0.004937f,  0.005012f,  0.000491f,  0.005498f,  0.000759f,  0.005970f,  -0.009263f, 0.011162f,\
  0.013674f,  -0.001277f, 0.005071f,  0.005010f,  0.000497f,  0.005523f,  0.000774f,  0.006005f,  -0.009319f, 0.011142f,\
  0.014016f,  -0.001505f, 0.005204f,  0.005005f,  0.000503f,  0.005545f,  0.000789f,  0.006038f,  -0.009371f, 0.011120f,\
  0.014353f,  -0.001733f, 0.005336f,  0.004999f,  0.000511f,  0.005565f,  0.000805f,  0.006068f,  -0.009420f, 0.011097f,\
  0.014685f,  -0.001961f, 0.005465f,  0.004990f,  0.000518f,  0.005581f,  0.000820f,  0.006096f,  -0.009465f, 0.011073f,\
  0.015010f,  -0.002187f, 0.005594f,  0.004979f,  0.000527f,  0.005596f,  0.000835f,  0.006121f,  -0.009507f, 0.011047f,\
  0.015330f,  -0.002412f, 0.005720f,  0.004966f,  0.000535f,  0.005608f,  0.000850f,  0.006144f,  -0.009547f, 0.011019f,\
  0.015644f,  -0.002636f, 0.005844f,  0.004951f,  0.000544f,  0.005617f,  0.000865f,  0.006166f,  -0.009583f, 0.010991f,\
  0.015951f,  -0.002858f, 0.005967f,  0.004935f,  0.000554f,  0.005625f,  0.000879f,  0.006185f,  -0.009617f, 0.010961f,\
  0.016252f,  -0.003077f, 0.006087f,  0.004916f,  0.000563f,  0.005630f,  0.000894f,  0.006202f,  -0.009647f, 0.010930f,\
  0.016546f,  -0.003295f, 0.006205f,  0.004897f,  0.000573f,  0.005634f,  0.000908f,  0.006217f,  -0.009675f, 0.010898f,\
  0.016834f,  -0.003511f, 0.006321f,  0.004875f,  0.000583f,  0.005635f,  0.000922f,  0.006230f,  -0.009700f, 0.010865f,\
  0.017114f,  -0.003723f, 0.006435f,  0.004853f,  0.000594f,  0.005634f,  0.000936f,  0.006241f,  -0.009723f, 0.010831f,\
  0.017388f,  -0.003933f, 0.006546f,  0.004829f,  0.000604f,  0.005632f,  0.000950f,  0.006251f,  -0.009743f, 0.010796f,\
  0.017654f,  -0.004140f, 0.006655f,  0.004804f,  0.000615f,  0.005628f,  0.000963f,  0.006259f,  -0.009761f, 0.010759f,\
  0.017914f,  -0.004343f, 0.006761f,  0.004778f,  0.000625f,  0.005623f,  0.000976f,  0.006266f,  -0.009776f, 0.010722f,\
  0.018166f,  -0.004543f, 0.006865f,  0.004751f,  0.000636f,  0.005616f,  0.000989f,  0.006270f,  -0.009789f, 0.010684f,\
  0.018410f,  -0.004740f, 0.006965f,  0.004723f,  0.000647f,  0.005608f,  0.001002f,  0.006274f,  -0.009800f, 0.010645f,\
  0.018647f,  -0.004932f, 0.007063f,  0.004695f,  0.000658f,  0.005598f,  0.001014f,  0.006276f,  -0.009808f, 0.010605f,\
  0.018877f,  -0.005121f, 0.007159f,  0.004665f,  0.000669f,  0.005587f,  0.001026f,  0.006276f,  -0.009815f, 0.010564f,\
  0.019098f,  -0.005305f, 0.007251f,  0.004635f,  0.000679f,  0.005576f,  0.001037f,  0.006276f,  -0.009819f, 0.010523f,\
  0.019313f,  -0.005486f, 0.007340f,  0.004605f,  0.000690f,  0.005562f,  0.001049f,  0.006274f,  -0.009822f, 0.010481f,\
  0.019519f,  -0.005662f, 0.007427f,  0.004574f,  0.000700f,  0.005548f,  0.001059f,  0.006270f,  -0.009822f, 0.010438f,\
  0.019718f,  -0.005833f, 0.007511f,  0.004542f,  0.000711f,  0.005533f,  0.001070f,  0.006266f,  -0.009821f, 0.010394f,\
  0.019908f,  -0.006000f, 0.007591f,  0.004510f,  0.000721f,  0.005517f,  0.001080f,  0.006260f,  -0.009818f, 0.010349f,\
  0.020092f,  -0.006163f, 0.007669f,  0.004478f,  0.000731f,  0.005501f,  0.001090f,  0.006254f,  -0.009813f, 0.010304f,\
  0.020267f,  -0.006321f, 0.007743f,  0.004446f,  0.000741f,  0.005483f,  0.001099f,  0.006246f,  -0.009806f, 0.010259f,\
  0.020435f,  -0.006474f, 0.007815f,  0.004413f,  0.000750f,  0.005465f,  0.001108f,  0.006237f,  -0.009798f, 0.010212f,\
  0.020595f,  -0.006622f, 0.007883f,  0.004381f,  0.000760f,  0.005446f,  0.001117f,  0.006228f,  -0.009788f, 0.010166f,\
  0.020747f,  -0.006766f, 0.007949f,  0.004348f,  0.000769f,  0.005426f,  0.001125f,  0.006217f,  -0.009777f, 0.010118f,\
  0.020892f,  -0.006905f, 0.008011f,  0.004315f,  0.000777f,  0.005406f,  0.001133f,  0.006206f,  -0.009764f, 0.010070f,\
  0.021029f,  -0.007039f, 0.008070f,  0.004282f,  0.000786f,  0.005386f,  0.001141f,  0.006193f,  -0.009749f, 0.010022f,\
  0.021159f,  -0.007169f, 0.008127f,  0.004250f,  0.000794f,  0.005364f,  0.001148f,  0.006180f,  -0.009733f, 0.009973f,\
  0.021281f,  -0.007293f, 0.008180f,  0.004217f,  0.000802f,  0.005343f,  0.001154f,  0.006166f,  -0.009716f, 0.009923f,\
  0.021396f,  -0.007414f, 0.008230f,  0.004184f,  0.000809f,  0.005321f,  0.001161f,  0.006151f,  -0.009697f, 0.009873f,\
  0.021504f,  -0.007529f, 0.008278f,  0.004151f,  0.000816f,  0.005298f,  0.001167f,  0.006136f,  -0.009678f, 0.009823f,\
  0.021605f,  -0.007640f, 0.008323f,  0.004119f,  0.000823f,  0.005275f,  0.001172f,  0.006119f,  -0.009656f, 0.009772f,\
  0.021699f,  -0.007746f, 0.008364f,  0.004086f,  0.000829f,  0.005252f,  0.001177f,  0.006102f,  -0.009634f, 0.009721f,\
  0.021786f,  -0.007847f, 0.008403f,  0.004054f,  0.000835f,  0.005228f,  0.001182f,  0.006085f,  -0.009610f, 0.009669f,\
  0.021866f,  -0.007944f, 0.008439f,  0.004022f,  0.000841f,  0.005204f,  0.001187f,  0.006066f,  -0.009585f, 0.009617f,\
  0.021939f,  -0.008037f, 0.008473f,  0.003990f,  0.000846f,  0.005180f,  0.001191f,  0.006047f,  -0.009559f, 0.009565f,\
  0.022006f,  -0.008125f, 0.008504f,  0.003958f,  0.000851f,  0.005155f,  0.001194f,  0.006027f,  -0.009532f, 0.009512f,\
  0.022066f,  -0.008209f, 0.008532f,  0.003926f,  0.000856f,  0.005130f,  0.001198f,  0.006007f,  -0.009503f, 0.009459f\
  };

const float as7341_array1[]={1,2,3};

void DelayUs(uint32_t us, TaskHandle_t *RefTask){
  xAS7341_DebaunceIntHandle = *RefTask;

  SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0);
  while (!SysCtlPeripheralReady(SYSCTL_PERIPH_TIMER0)); // Ensure it's ready

  TimerConfigure(TIMER0_BASE, TIMER_CFG_ONE_SHOT);

  // Set Load Value (Convert us to clock cycles)
  TimerLoadSet(TIMER0_BASE, TIMER_A, (SysCtlClockGet() / 1e6) * us - 1);

/*
TimerIntRegister(uint32_t ui32Base, uint32_t ui32Timer,
                 void (*pfnHandler)(void))
*/
  TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
  TimerIntRegister(TIMER0_BASE, TIMER_A, AS7341_Debaunce);
  IntMasterEnable();
  
  // Start Timer
  TimerEnable(TIMER0_BASE, TIMER_A);
  
  UARTprintf("\rDelay Init\n");
  xSemaphoreTake(xAS7341_DebaunceIntHandle, portMAX_DELAY); //SINT_MUX interruption
  UARTprintf("\rDelay Done\n");
  // Wait until timer expires
  //while (TimerValueGet(TIMER0_BASE, TIMER_A) != 0);

  //  Disable Timer After Completion
  TimerDisable(TIMER0_BASE, TIMER_A);

}

void AS7341_IntWatchDog(void* ptr){
  SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0);
  while (!SysCtlPeripheralReady(SYSCTL_PERIPH_TIMER0)); // Ensure it's ready

  TimerConfigure(TIMER0_BASE,TIMER_CFG_PERIODIC);
  
  uint32_t us = 2000;
  // Set Load Value (Convert us to clock cycles)
  TimerLoadSet(TIMER0_BASE, TIMER_A, (SysCtlClockGet() / 1e6) * us - 1);

/*
TimerIntRegister(uint32_t ui32Base, uint32_t ui32Timer,
                 void (*pfnHandler)(void))
*/
  TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
  TimerIntRegister(TIMER0_BASE, TIMER_A, AS7341_Debaunce);
  IntPrioritySet(INT_GPIOF, 0x05);
  IntMasterEnable();
  
  // Start Timer
  TimerEnable(TIMER0_BASE, TIMER_A);
  
  while(1){
    //UARTprintf("\t\t\tDelay Init\n");
    //GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, SLEEP_PIN);
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    //xSemaphoreGive(AS7341_Semphr);
    //GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, 0);
    UARTprintf("\t\t\t\t\t\t - AS7341 [WATCH DOG]\n");
    //xSemaphoreTake(xAS7341_DebaunceIntHandle, portMAX_DELAY); //SINT_MUX interruption
    // Wait until timer expires
    //while (TimerValueGet(TIMER0_BASE, TIMER_A) != 0);

    //  Disable Timer After Completion
    //TimerDisable(TIMER0_BASE, TIMER_A);
  }
}
#endif
