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

bool AS7341_i2cInit(){
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
  as7341_enable_t enable_reg;
  enable_reg.FDEN   = 0;
  enable_reg.SMUXEN = 1;
  enable_reg.WEN    = 1;
  enable_reg.SP_EN  = 0;
  enable_reg.PON    = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ENABLE, enable_reg.value)) return false;
  return true;
}
bool AS7341_DevivceConfig(){
  as7341_config_t config_reg;
  config_reg.LED_SEL  = 1;
  config_reg.INT_SEL  = 1;
  config_reg.INT_MODE = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CONFIG, config_reg.value)) return false;
 
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

  as7341_led_t led_reg;
  led_reg.LED_ACT   = 0;
  led_reg.LED_DRIVE = 0;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_LED, led_reg.value)) return false;

  as7341_intenab_t intenab_reg;
  intenab_reg.ASIEN   = 0;
  intenab_reg.SP_IEN  = 0;
  intenab_reg.F_IEN   = 1;
  intenab_reg.SIEN    = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_INTENAB, inenab_reg.value)) return false;
  
  as7341_control_t control_reg;
  control_reg.SP_MAN_AZ       = 0;
  control_reg.FIFO_CLR        = 0; //Talvez valha pena ter uma funcao so pra esse cmd
  control_reg.CLEAR_SAI_ACT   = 0;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CONTROL, control_reg.value)) return false;
  
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
  az_config_reg.AZ_NTH_ITERATION   = 255; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_AZ_CONFIG, az_config_reg.value)) return false;

  as7341_cfg8_t cfg8_reg;
  cfg8_reg.FIFO_TH   = 3; //
  cfg8_reg.FD_AGC    = 0; //
  cfg8_reg.SP_AGC    = 0; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG8, cfg8_reg.value)) return false;
  
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
  cfg6_reg.SMUX = 0; //
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
  fifo_map_reg.FIFO_WRITE_CH4_DATA   = 1; //
  fifo_map_reg.FIFO_WRITE_CH5_DATA   = 0; //
  fifo_map_reg.FIFO_WRITE_ASTATUS    = 1; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_FIFO_MAP, fifo_map_reg.value)) return false;
  
  as7341_fifo_cfg0_t fifo_cfg0_reg;
  fifo_cfg0_reg.FIFO_WRITE_FD   = 0; //
  if(!AS7341_SetAcessAndWrite(AS7341_REG_FIFO_CFG0, fifo_cfg0_reg.value)) return false;

  return true;
}

bool AS7341_DisableSpecMen(){
  as7341_enable_t enable_reg;
  if(!AS7341_SetAcessAndRead(AS7341_REG_ENABLE, &enable_reg.value)) return false;
  enable_reg.SP_EN = 0;
  if(!AS7341_write(AS7341_REG_ENABLE, enable_reg.value)) return false;

  return true
}
bool AS7341_PowerOff(){
  as7341_enable_t enable_reg;
  if(!AS7341_SetAcessAndRead(AS7341_REG_ENABLE, &enable_reg.value)) return false;
  enable_reg.PON = 0;
  if(!AS7341_write(AS7341_REG_ENABLE, enable_reg.value)) return false;

  return true;
}
bool AS7341_BankAcessSet(uint8_t RegAdd){
  uint8_t RegLevelNeeded = 0;
  if(RegAdd>=0x80) RegLevelNeeded = AS7341_BANK_HIGH_ACESS;
  else             RegLevelNeeded = AS7341_BANK_LOW_ACESS;

  if(BankAcessControlValue!=RegLevelNeeded){
    BankAcessControlValue = RegLevelNeeded;
    as7341_cfg0_t cfg0_reg;
    if(!AS7341_read(AS7341_REG_CFG0, &cfg1_reg.value)) return false;

    cfg1_reg.REG_BANK = BankAcessControlValue;
    if(!AS7341_write(AS7341_REG_CFG0, cfg1_reg.value)) return false;
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
  if(!AS7341_read(regAdd, *data)) return false;
  return true;
}

bool AS7341_SetSMUX(uint8_t* photoDiode, uint8_t* ADC_ID){

  if(!AS7341_DisableSpecMen()) return false;
  // Enable special interrupt and SMUX interrupt
  as7341_cfg9_t cfg9_reg;
  if(!AS7341_SetAcessAndRead(AS7341_REG_CFG9, &cfg9_reg.value)) return false;
  cfg9_reg.SIEN_SMUX = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_CFG9, cfg9_reg.value)) return false;
  
  as7341_intenab_t intenab_reg;
  if(!AS7341_SetAcessAndRead( AS7341_REG_INTENAB, &intenab_reg.value)) return false;
  intenab_reg.SIEN = 1;
  if(!AS7341_SetAcessAndWrite(AS7341_REG_INTENAB, inenab_reg.value))   return false;
  if(!AS7341_WriteI2cReg2SMUX_Sel()) return false;
  
  
  if(!AS7341_SetI2cRegSMUX(photoDiode, ADC_ID)) return false;
  
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
  if(!AS7341_SetAcessAndWrite(AS7341_REG_ENABLE, enable_reg.value)) return false;

  return true;
}


bool AS7341_SetI2cRegSMUX(uint8_t* photoDiode, uint8_t* ADC_ID){
  
  uint8_t data2Send;
  uint8_t RegAdd;
  // Definições especiais
  /*                      i2c Reg   |  IDs    |   PHOTO  |
                        ------------------------------------
                             0x5     11 e 10     F4 e F2              
                             0xE     29 e 28     F6 e F8             
                            0x10     33 e 32   GPIO e F1               
                            0x11     35 e 34     C2 e INT            
                            0x13     39 e 38   FLKR e NIR            
  */
  for(uint8_t index=0; index<18; index++){
    if(PHOTO_F1_1   ==photoDiode[index]){
      as7341_reg1 PixelID_2;
      PixelID_2.MUX_SEL = ADC_ID[index]<<4;
      RegAdd = 0x1;
      data2Send = PixelID_2.value;
    } 
    if(PHOTO_F3_1   ==photoDiode[index]){
      as7341_reg0 PixelID_1;
      PixelID_1.MUX_SEL = ADC_ID[index];
      RegAdd = 0x0;
      data2Send = PixelID_1.value;
    }
    if(PHOTO_F5_1   ==photoDiode[index]){
      as7341_reg9 PixelID_19;
      PixelID_19.MUX_SEL = ADC_ID[index]<<4;
      RegAdd = 0x9;
      data2Send = PixelID_19.value;
    }
    if(PHOTO_F7_1   ==photoDiode[index]){
      as7341_regA PixelID_20;
      PixelID_20.MUX_SEL = ADC_ID[index];
      RegAdd = 0xA;
      data2Send = PixelID_20.value;
    }
    if(PHOTO_F6_1   ==photoDiode[index]){
      as7341_reg4 PixelID_8;
      PixelID_8.MUX_SEL = ADC_ID[index];
      RegAdd = 0x4;
      data2Send = PixelID_8.value;
    }
    if(PHOTO_F8_1   ==photoDiode[index]){
      as7341_reg3 PixelID_7;
      PixelID_7.MUX_SEL = ADC_ID[index]<<4;
      RegAdd = 0x3;
      data2Send = PixelID_7.value;
    }
    if(PHOTO_F2_1   ==photoDiode[index]){
      as7341_regC PixelID_25;
      PixelID_25.MUX_SEL = ADC_ID[index]<<4;
      RegAdd = 0xC;
      data2Send = PixelID_25.value;
    }
    if(PHOTO_F4_1   ==photoDiode[index]){
      as7341_regD PixelID_26;
      PixelID_26.MUX_SEL = ADC_ID[index];
      RegAdd = 0xD;
      data2Send = PixelID_26.value;
    }
    if(PHOTO_F4_2   ==photoDiode[index]){
      // Escrita Especial
      as7341_reg5 PixelID_11;
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
    if(PHOTO_F8_2   ==photoDiode[index]){
      // Escrita Especial
      as7341_regE PixelID_28;
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
    if(PHOTO_F7_2   ==photoDiode[index]){
      as7341_reg7 PixelID_14;
      PixelID_14.MUX_SEL = ADC_ID[index];
      RegAdd = 0x7;
      data2Send = PixelID_14.value;
    }
    if(PHOTO_F5_2   ==photoDiode[index]){
      as7341_reg6 PixelID_13;
      PixelID_13.MUX_SEL = ADC_ID[index]<<4;
      RegAdd = 0x6;
      data2Send = PixelID_13.value;
    }
    if(PHOTO_F3_2   ==photoDiode[index]){
      as7341_regF PixelID_31;
      PixelID_31.MUX_SEL = ADC_ID[index]<<4;
      RegAdd = 0xF;
      data2Send = PixelID_31.value;
    }
    if(PHOTO_F1_2   ==photoDiode[index]){
      // Escrita Especial
      as7341_reg0x10 PixelID_32;
      PixelID_32.value = ADC_ID[index];
      RegAdd = 0x10;
      data2Send = PixelID_32.value;
    }
    if(PHOTO_CLEAR_1==photoDiode[index]){
      as7341_reg8 PixelID_17;
      PixelID_17.MUX_SEL = ADC_ID[index]<<4;
      RegAdd = 0x8;
      data2Send = PixelID_17.value;
    }  
    if(PHOTO_CLEAR_2==photoDiode[index]){
      // Escrita Especial
      as7341_reg0x11 PixelID_35;
      PixelID_35.value = ADC_ID[index];
      RegAdd = 0x11;
      data2Send = PixelID_35.value;
    }  
    if(PHOTO_NIR    ==photoDiode[index]){
      // Escrita Especial
      as7341_reg0x13 PixelID_38;
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
    if(DARK         ==photoDiode[index]){
      as7341_reg0x12 PixelID_37;
      PixelID_37.MUX_SEL = ADC_ID[index]<<4;
      RegAdd = 0x12;
      data2Send = PixelID_37.value;
    }

    if(!AS7341_write(regAdd, data2Send)) return false;
  }
}

#endif
