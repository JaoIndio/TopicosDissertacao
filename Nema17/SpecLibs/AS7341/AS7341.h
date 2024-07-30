// AS7341 Sensor Register Definitions
// Based on:
/*
  https://www.mouser.com/catalog/specsheets/AMS_03152019_AS7341_DS000504_1-00.pdf
*/
#ifndef AS7341_H
#define AS7341_H

// Register Addresses
#define AS7341_ADDR 0x39 // 7-bit I2C address for the AS7341

// Enable and Configuration Registers
#define AS7341_REG_ENABLE        0x80
#define AS7341_REG_ATIME         0x81
#define AS7341_REG_WTIME         0x83
#define AS7341_REG_SP_TH_L_LSB   0x84
#define AS7341_REG_SP_TH_L_MSB   0x85
#define AS7341_REG_SP_TH_H_LSB   0x86
#define AS7341_REG_SP_TH_H_MSB   0x87
#define AS7341_REG_CFG0          0xA9
#define AS7341_REG_CFG1          0xAA
#define AS7341_REG_CFG3          0xAC
#define AS7341_REG_CFG6          0xAF
#define AS7341_REG_CFG8          0xB1
#define AS7341_REG_CFG9          0xB2
#define AS7341_REG_CFG10         0xB3
#define AS7341_REG_CFG12         0xB5
#define AS7341_REG_PERS          0xBD
#define AS7341_REG_GPIO1         0x73
#define AS7341_REG_GPIO2         0xBE
#define AS7341_REG_CONTROL       0xFA
#define AS7341_REG_FIFO_MAP      0xFC
#define AS7341_REG_FIFO_LVL      0xFD
#define AS7341_REG_FDATA_L       0xFE
#define AS7341_REG_FDATA_H       0xFF

// Device Identification Registers
#define AS7341_REG_AUXID         0x90
#define AS7341_REG_REVID         0x91
#define AS7341_REG_ID            0x92

// Status Registers
#define AS7341_REG_STATUS        0x93
#define AS7341_REG_ASTATUS       0x94
#define AS7341_REG_STATUS2       0xA3
#define AS7341_REG_STATUS3       0xA4
#define AS7341_REG_STATUS5       0xA6
#define AS7341_REG_STATUS6       0xA7
#define AS7341_REG_FD_STATUS     0xDB

// Spectral Data Registers
#define AS7341_REG_CH0_DATA_L    0x95
#define AS7341_REG_CH0_DATA_H    0x96
#define AS7341_REG_CH1_DATA_L    0x97
#define AS7341_REG_CH1_DATA_H    0x98
#define AS7341_REG_CH2_DATA_L    0x99
#define AS7341_REG_CH2_DATA_H    0x9A
#define AS7341_REG_CH3_DATA_L    0x9B
#define AS7341_REG_CH3_DATA_H    0x9C
#define AS7341_REG_CH4_DATA_L    0x9D
#define AS7341_REG_CH4_DATA_H    0x9E
#define AS7341_REG_CH5_DATA_L    0x9F
#define AS7341_REG_CH5_DATA_H    0xA0



//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//********************** 10.2.1 Enable And Configuration Register **********************
// Bit Definitions for ENABLE Register (0x80)
#define AS7341_ENABLE_PON        (1 << 0) // Power ON
#define AS7341_ENABLE_SP_EN      (1 << 1) // Spectral Measurement Enable
#define AS7341_ENABLE_WEN        (1 << 3) // Wait Enable
#define AS7341_ENABLE_SMUXEN     (1 << 4) // SMUX Enable
#define AS7341_ENABLE_FDEN       (1 << 6) // Flicker Detection Enable

// Bit Definitions for CONFIG Register (0x70)
#define AS7341_CONFIG_LED_SEL    (1 << 3) // LED Control
#define AS7341_CONFIG_INT_SEL    (1 << 2) // Interrupt Selection
#define AS7341_CONFIG_INT_MODE_SPM     0 // Interrupt Mode Mask
#define AS7341_CONFIG_INT_MODE_SYNS    1 // Interrupt Mode Mask
#define AS7341_CONFIG_INT_MODE_SYND    3 // Interrupt Mode Mask

// Bit Definitions for GPIO Register (0x73)
#define AS7341_GPIO_REG_PD_GPIO   (1 << 1) //1: Photo diode connected to pin GPIO
#define AS7341_GPIO_REG_PD_INT    (1 << 0) //1: Photo diode connected to pin INT

// Bit Definitions for GPIO 2 Register (0xBE)
#define AS7341_GPIO2_REG_IN       (1 << 0) //GPIO Input. Indicates the status of the GPIO input if GPIO_IN_EN is set
#define AS7341_GPIO2_REG_OUT      (1 << 1) // GPIO Output. If set, the output state of the GPIO is active directly.
#define AS7341_GPIO2_REG_IN_EN    (1 << 2) // GPIO Input Enable. If set, the GPIO pin accepts a non-floating input.
#define AS7341_GPIO2_REG_INV      (1 << 3) //GPIO Invert. If set, the GPIO output is inverted.

// Bit Definitions for LED Register (0x74)
#define AS7341_LED_LED_ACT       (1 << 7) // LED Control
#define AS7341_LED_DRIVE_MASK  (0x7F << 0)    // LED Driving Strength Mask

// Bit Definitions for INTENAB Register (0xF9)
#define AS7341_INTENAB_ASIEN     (1 << 7) // Spectral and Flicker Detect Saturation Interrupt Enable
#define AS7341_INTENAB_SP_IEN    (1 << 3) // Spectral Interrupt Enable
#define AS7341_INTENAB_F_IEN     (1 << 2) // FIFO Buffer Interrupt Enable
#define AS7341_INTENAB_SIEN      (1 << 0) // System Interrupt Enable. When asserted permits system interrupts to be generated. Indicates that flicker detection status has changed or SMUX operation has finishedFIFO Buffer Interrupt Enable


// CONTROL Register (Address 0xFA)
#define AS7341_CONTROL_SP_MAN_AZ      (1 << 2) // Spectral Engine Manual Autozero.
#define AS7341_CONTROL_FIFO_CLR       (1 << 1) // FIFO Buffer Clear
#define AS7341_CONTROL_CLEAR_SAI_ACT  (1 << 0) // Clear Sleep-After-Interrupt Active



//********************** 10.2.2 ADC Timing Configuration/Integration Time **********************
// ATIME   Register (Address 0x81)
#define AS7341_ATIME_MAKS   (0xFF << 0) // Integration time

// ASTEP   Register (Address 0xCA, 0xCB)
#define AS7341_ASTEP_LSB   (0xFF << 0) // Integration time step size
#define AS7341_ASTEP_MSB   (0xFF << 0) // Integration time step size

// WTIME   Register (Address 0x83)
#define AS7341_WTIME   (0xFF << 0) // Spectral Measurement Wait time

// ITIME   Register (Address 0x63, 0x64, 0x65)
#define AS7341_ITIME_L   (0xFF << 0) //
#define AS7341_ITIME_M   (0xFF << 0) //
#define AS7341_ITIME_H   (0xFF << 0) //

// EDGE    Register (Address 0x72)
#define AS7341_EDGE_SYNC   (1 << 1) // Number of falling SYNC-edges between start and stop of integration in mode SYND

// FD_TIME Register (Address 0xD8)
#define AS7341_FD_TIME_LSB_MASK (0xFF << 0) //

// FD_TIME Register (Address 0xDA)
#define AS7341_FD_TIME_MSB_MASK  7//
#define AS7341_FD_GAIN_MASK xF8 //
#define AS7341_FD_GAIN_SHIFT 3 //
#define AS7341_FD_GAIN_VALUE(x) ((x<<AS7341_FD_TIME_FD_GAIN_SHIFT) & AS7341_FD_GAIN_MASK )




//********************** 10.2.3 ADC Configuration (gain, AGC…) **********************
//CFG1         Register (Address 0xAA)
#define AS7341_CFG1_AGAIN_MASK 0xF      // Spectral engines gain setting.

//CFG10        Register (Address 0xB3)
#define AS7341_CFG10_AGC_H_MASK (3 << 6)      
#define AS7341_CFG10_AGC_H_SHIFT 6       
#define AS7341_CFG10_AGC_H_VALUE(x) ((x<<AS7341_CFG10_AGC_H_SHIFT) & AS7341_CFG10_AGC_H_MASK) //AGC High Hysteresis
#define AS7341_CFG10_AGC_L_MASK (3 << 4)       
#define AS7341_CFG10_AGC_L_SHIFT 4       
#define AS7341_CFG10_AGC_L_VALUE(x) ((x<<AS7341_CFG10_AGC_L_SHIFT) & AS7341_CFG10_AGC_L_MASK) //AGC Low Hysteresis
#define AS7341_CFG10_FD_PERS_MASK 3 //Flicker Detect Persistence.     

//AZ_CONFIG    Register (Address 0xD6)
#define AS7341_AZ_CONFIG_AZ_NTH_ITERATION_MASK 0xFF //AUTOZERO FREQUENCY.   

//AGC_GAIN_MAX Register (Address 0xCF)
#define AS7341_AGC_GAIN_MAX_AGC_FD_GAIN_MAX_MASK  0xF << 4 
#define AS7341_AGC_GAIN_MAX_AGC_FD_GAIN_MAX_SHIFT 4
#define AS7341_AGC_GAIN_MAX_AGC_FD_GAIN_MAX_VALUE(x) ((x<<AS7341_AGC_GAIN_MAX_AGC_FD_GAIN_MAX_SHIFT) & AS7341_AGC_GAIN_MAX_AGC_FD_GAIN_MAX_MASK) //Flicker Detection AGC Gain Max.
#define AS7341_AGC_GAIN_MAX_AGC_AGAIN_MAX_MASK  0xF //AGC Gain Max

//CFG8         Register (Address 0xB1)
#define AS7341_CFG8_FIFO_TH_MASK (3 << 6) //FIFO Threshold       
#define AS7341_CFG8_FIFO_TH_SHIFT 6 //FIFO Threshold       
#define AS7341_CFG8_FIFO_TH_VALUE(x) ((x<<AS7341_CFG8_FIFO_TH_SHIFT) & AS7341_CFG8_FIFO_TH_MASK) //FIFO Threshold.
#define AS7341_CFG8_FD_AGC  (1 << 3) //Flicker Detect AGC Enable.     
#define AS7341_CFG8_SP_AGC  (1 << 2) //Spectral AGC enable      




//********************** 10.2.4 Device Identification **********************
//AUXID Register (Address 0x90)
#define AS7341_AUXID

//REVID Register (Address 0x91)
#define AS7341_REVID

//ID    Register (Address 0x92)
#define AS7341_ID   




//********************** 10.2.5 Spectral Interrupt Configuration **********************
//SP_TH_L_LSB Register (Address 0x84)
#define AS7341_SP_TH_L_LSB

//SP_TH_L_MSB Register (Address 0x85)
#define AS7341_SP_TH_L_MSB

//SP_TH_H_LSB Register (Address 0x86)
#define AS7341_SP_TH_H_LSB

//SP_TH_H_MSB Register (Address 0x87)
#define AS7341_SP_TH_H_MSB

//CFG12       Register (Address 0xB5)
#define AS7341_CFG12      




//********************** 10.2.6 Device Status Register **********************
//STAT      Register (Address 0x71)
#define AS7341_STAT     

//STATUS    Register (Address 0x93)
#define AS7341_STATUS   

//STATUS2   Register (Address 0xA3)
#define AS7341_STATUS2  

//STATUS3   Register (Address 0xA4)
#define AS7341_STATUS3  

//STATUS5   Register (Address 0xA6)
#define AS7341_STATUS5  

//STATUS6   Register (Address 0xA7)
#define AS7341_STATUS6  

//FD_STATUS Register (Address 0xDB)
#define AS7341_FD_STATUS




//********************** 10.2.7 Spectral Data and Status **********************
//ASTATUS  Register (Address 0x60 or 0x94)
#define AS7341_ASTATUS 

//CH0_DATA Register (Address 0x95/0x96)
#define AS7341_CH0_DATA

//CH1_DATA Register (Address 0x97/0x98)
#define AS7341_CH1_DATA

//CH2_DATA Register (Address 0x99/0x9A)
#define AS7341_CH2_DATA

//CH3_DATA Register (Address 0x9B/0x9C
#define AS7341_CH3_DATA

//CH4_DATA Register (Address 0x9D/0x9E)
#define AS7341_CH4_DATA

//CH5_DATA Register (Address 0x9F/0xA0)
#define AS7341_CH5_DATA




//********************** 10.2.8 Miscellaneous Configuration **********************
//CFG0 Register (Address 0xA9)
#define AS7341_CFG0

//CFG3 Register (Address 0xAC)
#define AS7341_CFG3

//CFG6 Register (Address 0xAF)
#define AS7341_CFG6

//CFG9 Register (Address 0xB2)
#define AS7341_CFG9

//PERS Register (Address 0xBD)
#define AS7341_PERS




//********************** 10.2.9 FIFO Buffer Data and Status **********************
//FIFO_MAP  Register (Address 0xFC)
#define AS7341_FIFO_MAP 

//FIFO_CFG0 Register (Address 0xD7)
#define AS7341_FIFO_CFG0

//FIFO_LVL  Register (Address 0xFD)
#define AS7341_FIFO_LVL 

//FDATA     Register (Address 0xFE and 0xFF)
#define AS7341_FDATA    




// Status Bits in STATUS Register (0x93)
#define AS7341_STATUS_ASAT       (1 << 7) // Spectral and Flicker Detect saturation
#define AS7341_STATUS_AINT       (1 << 3) // Spectral Channel Interrupt
#define AS7341_STATUS_FINT       (1 << 2) // FIFO Buffer Interrupt
#define AS7341_STATUS_CINT       (1 << 1) // Calibration Interrupt
#define AS7341_STATUS_SINT       (1 << 0) // System Interrupt




//-------------------------------------------
//-------------------------------------------
//-------------------------------------------
bool AS7341_write();
bool AS7341_read();
bool AS7341_Enable();
bool AS7341_DevivceConfig();
bool AS7341_ADC_TimingConfig();
bool AS7341_ADC_Config();
bool AS7341_InterruptionConfig();
bool AS7341_DeviceStatus();
bool AS7341_SpecData();
bool AS7341_SpecStatus();
bool AS7341_OtherConfig();
bool AS7341_BufferData();
bool AS7341_BufferConfig();
#endif // AS7341_H

