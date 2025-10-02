

/*
  https://www.mouser.com/catalog/specsheets/AMS_03152019_AS7341_DS000504_1-00.pdf
  https://dfimg.dfrobot.com/nobody/wiki/6a2a00069245ca5b43dd2a7e7c35f831.pdf
  https://look.ams-osram.com/m/2a3e700eb3b0a0cf/original/AS7341_UG000400_6-00.pdf
  https://www.dfrobot.com/product-2131.html
  https://wiki.dfrobot.com/Gravity_AS7341_Visible_Light_Sensor_SKU_SEN0364

  /home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/gitFiles/Documents/AS7341/AS7341_AN000666_1-00.pdf
*/
#ifndef AS7341_H
#define AS7341_H

// Register Addresses
#define AS7341_ADDR 0x39 // 7-bit I2C address for the AS7341

//********************** 10.2.1 Enable And Configuration Register **********************
#define AS7341_REG_ENABLE        0x80
#define AS7341_REG_CONFIG        0x70
#define AS7341_REG_GPIO1         0x73
#define AS7341_REG_GPIO2         0xBE
#define AS7341_REG_LED           0x74
#define AS7341_REG_INTENAB       0xF9
#define AS7341_REG_CONTROL       0xFA


//********************** 10.2.2 ADC Timing Configuration/Integration Time **********************
#define AS7341_REG_ATIME         0x81
#define AS7341_REG_ASTEP_L       0xCA
#define AS7341_REG_ASTEP_H       0xCB
#define AS7341_REG_WTIME         0x83
#define AS7341_REG_ITIME_L       0x63
#define AS7341_REG_ITIME_M       0x64
#define AS7341_REG_ITIME_H       0x65
#define AS7341_REG_EDGE          0x72
#define AS7341_REG_FD_TIME_LSB   0xD8
#define AS7341_REG_FD_TIME       0xDA



//********************** 10.2.3 ADC Configuration (gain, AGC…) **********************
#define AS7341_REG_CFG1          0xAA
#define AS7341_REG_CFG10         0xB3
#define AS7341_REG_AZ_CONFIG     0xD6
#define AS7341_REG_AGC_GAIN_MAX  0xCF
#define AS7341_REG_CFG8          0xB1


//********************** 10.2.4 Device Identification **********************
#define AS7341_REG_AUXID         0x90
#define AS7341_REG_REVID         0x91
#define AS7341_REG_ID            0x92



//********************** 10.2.5 Spectral Interrupt Configuration **********************
#define AS7341_REG_SP_TH_L_LSB   0x84
#define AS7341_REG_SP_TH_L_MSB   0x85
#define AS7341_REG_SP_TH_H_LSB   0x86
#define AS7341_REG_SP_TH_H_MSB   0x87
#define AS7341_REG_CFG12         0xB5


//********************** 10.2.6 Device Status Register **********************
#define AS7341_REG_STAT          0x71
#define AS7341_REG_STATUS        0x93
#define AS7341_REG_STATUS2       0xA3
#define AS7341_REG_STATUS3       0xA4
#define AS7341_REG_STATUS5       0xA6
#define AS7341_REG_STATUS6       0xA7
#define AS7341_REG_FD_STATUS     0xDB


//********************** 10.2.7 Spectral Data and Status **********************
#define AS7341_REG_ASTATUS1       0x60
#define AS7341_REG_ASTATUS2       0x94
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


//********************** 10.2.8 Miscellaneous Configuration **********************
#define AS7341_REG_CFG0          0xA9
#define AS7341_REG_CFG3          0xAC
#define AS7341_REG_CFG6          0xAF
#define AS7341_REG_CFG9          0xB2
#define AS7341_REG_PERS          0xBD


//********************** 10.2.9 FIFO Buffer Data and Status **********************
#define AS7341_REG_FIFO_MAP      0xFC
#define AS7341_REG_FIFO_CFG0     0xD7
#define AS7341_REG_FIFO_LVL      0xFD
#define AS7341_REG_FDATA_L       0xFE
#define AS7341_REG_FDATA_H       0xFF


#define AS7341_BANK_HIGH_ACESS  0
#define AS7341_BANK_LOW_ACESS   1


#define PHOTO_F1_1                2
#define PHOTO_F3_1                1
#define PHOTO_F5_1                19
#define PHOTO_F7_1                20
#define PHOTO_F6_1                8
#define PHOTO_F8_1                7
#define PHOTO_F2_1                25
#define PHOTO_F4_1                26

#define PHOTO_F4_2                11
#define PHOTO_F2_2                10
#define PHOTO_F8_2                28
#define PHOTO_F6_2                29
#define PHOTO_F7_2                14
#define PHOTO_F5_2                13
#define PHOTO_F3_2                31
#define PHOTO_F1_2                32


#define PHOTO_CLEAR_1             17
#define PHOTO_CLEAR_2             35
#define PHOTO_NIR                 38
#define PHOTO_FLICKER             39
#define GPIO_INPUT                33
#define INT_INPUT                 34
#define DARK                      37

#define CONNECT_TO_GND            0
#define CONNECT_TO_ADC0           1
#define CONNECT_TO_ADC1           2
#define CONNECT_TO_ADC2           3
#define CONNECT_TO_ADC3           4
#define CONNECT_TO_ADC4           5
#define CONNECT_TO_ADC5           6

//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------
//********************** 10.2.1 Enable And Configuration Register **********************
// Bit Definitions for ENABLE Register (0x80)
typedef union{
  struct{
    uint8_t PON : 1;        ///< Bit 0: Power ON (0: Desabilitado, 1: Habilitado)
    uint8_t SP_EN : 1;      ///< Bit 1: Habilitação de Medição Espectral (0: Desabilitado, 1: Habilitado)
    uint8_t : 1;            ///< Bit 2: Reservado
    uint8_t WEN : 1;        ///< Bit 3: Habilitação de Espera (0: Desabilitado, 1: Habilitado)
    uint8_t SMUXEN : 1;     ///< Bit 4: Habilitação de SMUX (1: Inicia comando SMUX)
    uint8_t reserved_5 : 1; ///< Bit 5: Reservado
    uint8_t FDEN : 1;       ///< Bit 6: Habilitação de Detecção de Cintilação (0: Desabilitado, 1: Habilitado)
    uint8_t : 1;            ///< Bit 7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_enable_t;

// Bit Definitions for CONFIG Register (0x70)
typedef union{
  struct{
    uint8_t INT_MODE : 2; ///< Bits 0 a 1: Modo de detecção de luz ambiente (0: SPM, 1: SYNS, 2: Reservado, 3: SYND)
    uint8_t INT_SEL : 1;  ///< Bit 2: Seleção de sincronização (0: Desabilitado, 1: Habilitado)
    uint8_t LED_SEL : 1;  ///< Bit 3: Controle do LED (0: LED externo não controlado pelo AS7341, 1: LED externo conectado ao pino LDR)
    uint8_t : 4;          ///< Bits 4 a 7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_config_t;

// Bit Definitions for GPIO Register (0x73)
typedef union{
  struct{
    uint8_t PD_INT : 1;  ///< Bit 0: Fotodiodo conectado ao pino INT (0: Desconectado, 1: Conectado)
    uint8_t PD_GPIO : 1; ///< Bit 1: Fotodiodo conectado ao pino GPIO (0: Desconectado, 1: Conectado)
    uint8_t : 6;         ///< Bits 2 a 7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_gpio_t;

// Bit Definitions for GPIO 2 Register (0xBE)
typedef union{
  struct{
    uint8_t GPIO_IN : 1;    ///< Bit 0: Entrada GPIO (0: Estado de entrada do GPIO)
    uint8_t GPIO_OUT : 1;   ///< Bit 1: Saída GPIO (1: Estado de saída ativo diretamente)
    uint8_t GPIO_IN_EN : 1; ///< Bit 2: Habilitação de entrada GPIO (1: Pino GPIO aceita uma entrada não flutuante)
    uint8_t GPIO_INV : 1;   ///< Bit 3: Inversão de GPIO (1: Saída GPIO é invertida)
    uint8_t : 4;            ///< Bits 4 a 7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_gpio2_t;

// Bit Definitions for LED Register (0x74)
typedef union{
  struct{
    uint8_t LED_DRIVE : 6; ///< Bits 5 a 0: Força de condução do LED (000000: 4mA, 000001: 6mA, 000010: 8mA, 000011: 10mA, 000100: 12mA, ..., 111110: 256mA, 111111: 258mA)
    uint8_t : 1;           ///< Bit 6: Reservado
    uint8_t LED_ACT : 1;   ///< Bit 7: Controle de LED (0: LED externo desconectado do pino LDR, 1: LED externo conectado ao pino LDR)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_led_t;

// Bit Definitions for INTENAB Register (0xF9)
typedef union{
  struct{
    uint8_t SIEN : 1;   ///< Bit 0: Habilitação de Interrupção do Sistema (0: Desabilitado, 1: Habilitado)
    uint8_t : 1;        ///< Bit 1: Reservado
    uint8_t F_IEN : 1;  ///< Bit 2: Habilitação de Interrupção do Buffer FIFO (0: Desabilitado, 1: Habilitado)
    uint8_t SP_IEN : 1; ///< Bit 3: Habilitação de Interrupção Espectral (0: Desabilitado, 1: Habilitado)
    uint8_t : 3;        ///< Bits 4 a 6: Reservados
    uint8_t ASIEN : 1;  ///< Bit 7: Habilitação de Interrupção de Saturação Espectral e de Detecção de Cintilação (0: Desabilitado, 1: Habilitado)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_intenab_t;

// CONTROL Register (Address 0xFA)
typedef union{
  struct{
    uint8_t CLEAR_SAI_ACT : 1; ///< Bit 0: Limpeza do SAI Após Interrupção Ativa (0: Desabilitado, 1: Habilitado)
    uint8_t FIFO_CLR : 1;      ///< Bit 1: Limpeza do Buffer FIFO (0: Desabilitado, 1: Habilitado)
    uint8_t SP_MAN_AZ : 1;     ///< Bit 2: Autozero Manual do Motor Espectral (0: Desabilitado, 1: Habilitado)
    uint8_t : 5;               ///< Bits 3 a 7: Reservados
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_control_t;



//********************** 10.2.2 ADC Timing Configuration/Integration Time **********************
// ATIME   Register (Address 0x81)
typedef union{
  struct{
    uint8_t ATIME : 8; ///< Bits 0 a 7: Tempo de Integração (1 a 256 passos de integração)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_atime_t;

// ASTEP   Register (Address 0xCA, 0xCB)
typedef union{
  struct{
    uint16_t ASTEP : 16; ///< Bits 0 a 15: Tamanho do passo de tempo de integração (define o tamanho do passo em incrementos de 2,78µs)
  };
  struct{
    uint8_t ASTEP_L : 8; ///< Bits 0 a 7 do registrador 0xCA
    uint8_t ASTEP_H : 8; ///< Bits 8 a 15 do registrador 0xCB
  };
  uint16_t value; ///< Valor bruto do registrador
} as7341_astep_t;

// WTIME   Register (Address 0x83)
typedef union{
  struct{
    uint8_t WTIME : 8; ///< Bits 0 a 7: Tempo de Espera (define o atraso entre duas medições espectrais consecutivas)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_wtime_t;

// ITIME   Register (Address 0x63, 0x64, 0x65)
typedef union{
  struct{
    uint32_t ITIME_L : 8;  ///< Bits 0 a 7 do registrador 0x63: Tempo de Integração (byte inferior)
    uint32_t ITIME_M : 8;  ///< Bits 8 a 15 do registrador 0x64: Tempo de Integração (byte médio)
    uint32_t ITIME_H : 8;  ///< Bits 16 a 23 do registrador 0x65: Tempo de Integração (byte superior)
    uint32_t reserved : 8; ///< Bits 24 a 31: Reservado (não usado)
  };
  struct
  {
    uint8_t bytes[3]; ///< Acesso ao valor como uma matriz de 3 bytes
  };
  uint32_t value; ///< Valor bruto do registrador (24 bits)
} as7341_itime_t;

// EDGE    Register (Address 0x72)
typedef union{
  struct{
    uint8_t SYNC_EDGE : 8; ///< Bits 0 a 7: Número de bordas de sincronização SYNC (SYNC_EDGE + 1)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_edge_t;

// FD_TIME Register (Address 0xD8)
typedef union{
  struct{
    uint8_t FD_TIME_L;     ///< Bits 0 a 7 do registrador 0xD8: LSB do tempo de integração da detecção de cintilação
    uint8_t FD_TIME_H : 3; ///< Bits 0 a 2 do registrador 0xDA: MSB do tempo de integração da detecção de cintilação
    uint8_t FD_GAIN : 5;   ///< Bits 3 a 7 do registrador 0xDA: Ganho de Detecção de Cintilação
  };
  uint16_t value; ///< Valor bruto do registrador
} as7341_fd_time_t;

// FD_TIME Register (Address 0xDA)
#define AS7341_FD_TIME_MSB_MASK  7//
#define AS7341_FD_GAIN_MASK 0xF8 //
#define AS7341_FD_GAIN_SHIFT 3 //
#define AS7341_FD_GAIN_VALUE(x) ((x<<AS7341_FD_TIME_FD_GAIN_SHIFT) & AS7341_FD_GAIN_MASK )




//********************** 10.2.3 ADC Configuration (gain, AGC…) **********************
//CFG1         Register (Address 0xAA)
typedef union{
  struct{
    uint8_t AGAIN : 5; ///< Bits 0 a 4: Configuração do ganho dos motores espectrais (0.5x, 1x, 2x, 4x, 8x, 16x, 32x, 64x, 128x, 256x, 512x)
    uint8_t : 3;       ///< Bits 5 a 7: Reservados
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_cfg1_t;

//CFG10        Register (Address 0xB3)
typedef union{
  struct{
    uint8_t FD_PERS : 3; ///< Bits 0 a 2: Persistência de Detecção de Cintilação (0 a 7)
    uint8_t : 1;         ///< Bit 3: Reservado
    uint8_t AGC_L : 2;   ///< Bits 4 a 5: Histerese Baixa do AGC
    uint8_t AGC_H : 2;   ///< Bits 6 a 7: Histerese Alta do AGC
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_cfg10_t; 

//AZ_CONFIG    Register (Address 0xD6)
typedef union{
  struct{
    uint8_t AZ_NTH_ITERATION : 8; ///< Bits 0 a 7: Frequência de Autozero (0 a 255)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_az_config_t;  

//AGC_GAIN_MAX Register (Address 0xCF)
typedef union{
  struct{
    uint8_t AGC_AGAIN_MAX : 4;   ///< Bits 0 a 3: Ganho máximo do AGC para motores espectrais (0 a 15)
    uint8_t AGC_FD_GAIN_MAX : 4; ///< Bits 4 a 7: Ganho máximo do AGC para detecção de cintilação (0 a 15)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_agc_gain_max_t;

//CFG8         Register (Address 0xB1)
typedef union{
  struct{
    uint8_t : 1;         ///<
    uint8_t : 1;         ///<
    uint8_t SP_AGC : 1;  ///< Bit 7: Habilitação do AGC para motores espectrais     
    uint8_t FD_AGC : 1;  ///< Bit 6: Habilitação do AGC para detecção de cintilação 
    uint8_t : 2;         ///< Bits 2 a 5: Reservados
    uint8_t FIFO_TH : 2; ///< Bits 0 a 1: Limiar do FIFO (0 a 3)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_cfg8_t;  




//********************** 10.2.4 Device Identification **********************
//AUXID Register (Address 0x90)
typedef union{
    struct{
      uint8_t AUXID : 4; ///< Bits 0 a 3: Identificação auxiliar (somente leitura)
      uint8_t : 4;       ///< Bits 4 a 7: Reservados
    };
    uint8_t value; ///< Valor bruto do registrador
} as7341_auxid_t;

//REVID Register (Address 0x91)
typedef union{
  struct{
    uint8_t REV_ID : 3; ///< Bits 0 a 2: Identificação do número de revisão (somente leitura)
    uint8_t : 5;        ///< Bits 3 a 7: Reservados
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_revid_t;

//ID    Register (Address 0x92)
typedef union{
  struct{
    uint8_t : 2;
    uint8_t ID : 6;
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_id_t;




//********************** 10.2.5 Spectral Interrupt Configuration **********************
//SP_TH_L_LSB Register (Address 0x84 and 0x85)
typedef union{
  struct{
    uint8_t SP_TH_L_LSB; ///< Registrador 0x84: Byte menos significativo do threshold
    uint8_t SP_TH_L_MSB; ///< Registrador 0x85: Byte mais significativo do threshold
  };
  uint16_t value; ///< Valor bruto dos registradores (16 bits)
} as7341_sp_th_l_t;


//SP_TH_H_LSB Register (Address 0x86 and 0x87)
typedef union{
  struct{
    uint8_t SP_TH_H_LSB; ///< Registrador 0x86: Byte menos significativo do threshold
    uint8_t SP_TH_H_MSB; ///< Registrador 0x87: Byte mais significativo do threshold
  };
  uint16_t value; ///< Valor bruto dos registradores (16 bits)
} as7341_sp_th_h_t;


//CFG12       Register (Address 0xB5)
typedef union{
  struct{
    uint8_t SP_TH_CH : 3; ///< Bits 0 a 2: Canal de limiar espectral (0 a 4)
    uint8_t : 5;          ///< Bits 3 a 7: Reservados
  };
    uint8_t value; ///< Valor bruto do registrador
} as7341_cfg12_t;    




//********************** 10.2.6 Device Status Register **********************
//STAT      Register (Address 0x71)
typedef union{
  struct{
    uint8_t READY : 1;     ///< Bit 0: Status da medição espectral (0: Ocupado, 1: Pronto)
    uint8_t WAIT_SYNC : 1; ///< Bit 1: Espera por pulso de sincronização no GPIO para iniciar a integração (1: Esperando, 0: Não esperando)
    uint8_t : 6;           ///< Bits 2 a 7: Reservados
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_stat_t;    

//STATUS    Register (Address 0x93)
typedef union{
  struct{
    uint8_t SINT : 1;         ///< Bit 0: Interrupção do Sistema (0: Não setado, 1: Setado)
    uint8_t C_INT : 1;        ///< Bit 1: Interrupção de Calibração (0: Não setado, 1: Setado)
    uint8_t FINT : 1;         ///< Bit 2: Interrupção do Buffer FIFO (0: Não setado, 1: Setado)
    uint8_t AINT : 1;         ///< Bit 3: Interrupção do Canal Espectral (0: Não setado, 1: Setado)
    uint8_t reserved_6_4 : 3; ///< Bits 4 a 6: Reservados
    uint8_t ASAT : 1;         ///< Bit 7: Saturação Espectral e de Detecção de Cintilação (0: Não setado, 1: Setado)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_status_t;  

//STATUS2   Register (Address 0xA3)
typedef union{
  struct{
    uint8_t FDSAT_DIGITAL : 1; ///< Bit 0: Saturação Digital de Detecção de Cintilação (0: Não setado, 1: Setado)
    uint8_t FDSAT_ANALOG : 1;  ///< Bit 1: Saturação Analógica de Detecção de Cintilação (0: Não setado, 1: Setado)
    uint8_t : 1;               ///< Bit 2: Reservado
    uint8_t ASAT_ANALOG : 1;   ///< Bit 3: Saturação Analógica (0: Não setado, 1: Setado)
    uint8_t ASAT_DIGITAL : 1;  ///< Bit 4: Saturação Digital (0: Não setado, 1: Setado)
    uint8_t : 1;               ///< Bit 5: Reservado
    uint8_t AVALID : 1;        ///< Bit 6: Validade Espectral (0: Não válido, 1: Válido)
    uint8_t : 1;               ///< Bit 7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_status2_t;

//STATUS3   Register (Address 0xA4)
typedef union{
  struct{
    uint8_t : 4;          ///< Bits 0 a 3: Reservados
    uint8_t INT_SP_L : 1; ///< Bit 4: Interrupção Espectral Baixa (0: Não ocorreu, 1: Ocorreu)
    uint8_t INT_SP_H : 1; ///< Bit 5: Interrupção Espectral Alta (0: Não ocorreu, 1: Ocorreu)
    uint8_t : 2;          ///< Bits 6 a 7: Reservados
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_status3_t; 

//STATUS5   Register (Address 0xA6)
typedef union{
  struct{
    uint8_t : 2;           ///< Bits 0 a 1: Reservados
    uint8_t SINT_SMUX : 1; ///< Bit 2: Interrupção de Operação SMUX (0: Não ocorreu, 1: Ocorreu)
    uint8_t SINT_FD : 1;   ///< Bit 3: Interrupção de Detecção de Cintilação (0: Não ocorreu, 1: Ocorreu)
    uint8_t : 4;           ///< Bits 4 a 7: Reservados
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_status5_t;

//STATUS6   Register (Address 0xA7)
typedef union{
  struct{
    uint8_t INT_BUSY : 1;   ///< Bit 0: Inicialização Ocupada (0: Não ocupado, 1: Ocupado)
    uint8_t SAI_ACTIVE : 1; ///< Bit 1: Sono após Interrupção Ativa (0: Não ativo, 1: Ativo)
    uint8_t SP_TRIG : 1;    ///< Bit 2: Erro de Trigger Espectral (0: Não ocorreu, 1: Ocorreu)
    uint8_t : 1;            ///< Bit 3: Reservado
    uint8_t FD_TRIG : 1;    ///< Bit 4: Erro de Trigger de Detecção de Cintilação (0: Não ocorreu, 1: Ocorreu)
    uint8_t OVTEMP : 1;     ///< Bit 5: Detecção de Alta Temperatura (0: Temperatura normal, 1: Alta temperatura)
    uint8_t : 1;            ///< Bit 6: Reservado
    uint8_t FIFO_OV : 1;    ///< Bit 7: Estouro do Buffer FIFO (0: Não ocorreu, 1: Ocorreu)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_status6_t; 

//FD_STATUS Register (Address 0xDB)
typedef union{
  struct{
    uint8_t FD_100HZ_FLICKER : 1;       ///< Bit 0: Cintilação detectada a 100Hz (0: Não detectada, 1: Detectada)
    uint8_t FD_120HZ_FLICKER : 1;       ///< Bit 1: Cintilação detectada a 120Hz (0: Não detectada, 1: Detectada)
    uint8_t FD_100HZ_FLICKER_VALID : 1; ///< Bit 2: Detecção de Cintilação a 100Hz Válida (0: Não válida, 1: Válida)
    uint8_t FD_120HZ_FLICKER_VALID : 1; ///< Bit 3: Detecção de Cintilação a 120Hz Válida (0: Não válida, 1: Válida)
    uint8_t FD_SATURATION_DETECTED : 1; ///< Bit 4: Saturação detectada durante a medição de cintilação (0: Não detectada, 1: Detectada)
    uint8_t FD_MEASUREMENT_VALID : 1;   ///< Bit 5: Medição de Detecção de Cintilação Válida (0: Não válida, 1: Válida)
    uint8_t : 2;                        ///< Bits 6 a 7: Reservados
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_fd_status_t;




//********************** 10.2.7 Spectral Data and Status **********************
//ASTATUS  Register (Address 0x60 or 0x94)
typedef union{
  struct{
    uint8_t AGAIN_STATUS : 4; ///< Bits 0 a 3: Status do Ganho (0 a 15)
    uint8_t : 3;              ///< Bits 4 a 6: Reservados
    uint8_t ASAT_STATUS : 1;  ///< Bit 7: Status de Saturação (0: Não saturado, 1: Saturado)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_astatus_t;

//CH0_DATA Register (Address 0x95/0x96, 0x97/0x98, 0x99/0x9A, 0x9B/0x9C, 0x9D/0x9E, 0x9F/0xA0)
typedef union{
  struct{
    uint8_t DATA_L; ///< Byte inferior dos dados ADC
    uint8_t DATA_H; ///< Byte superior dos dados ADC
  };
  uint16_t value; ///< Valor de 16 bits dos dados ADC
} as7341_adc_data_t;





//********************** 10.2.8 Miscellaneous Configuration **********************
//CFG0 Register (Address 0xA9)
typedef union{
  struct{
    uint8_t : 2;           ///< Bits 0-1: Reservado
    uint8_t WLONG : 1;     ///< Bit 2: Trigger Longo (0: Normal, 1: Aumenta o WTIME por um fator de 16)
    uint8_t : 1;           ///< Bit 3: Reservado
    uint8_t REG_BANK : 1;  ///< Bit 4: Acesso ao Banco de Registradores (0: 0x80 e acima, 1: 0x60 a 0x74)
    uint8_t LOW_POWER : 1; ///< Bit 5: Baixa Potência (0: Desligado, 1: Ligado)
    uint8_t : 2;           ///< Bits 6-7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_cfg0_t;

//CFG3 Register (Address 0xAC)
typedef union{
  struct{
    uint8_t : 4;     ///< Bits 0-3: Reservado (0xC)
    uint8_t SAI : 1; ///< Bit 4: Sono após Interrupção (0: Desligado, 1: Ligado)
    uint8_t : 3;     ///< Bits 5-7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_cfg3_t;

//CFG6 Register (Address 0xAF)
typedef union{
  struct{
    uint8_t : 3;          ///< Bits 0-2: Reservado
    uint8_t SMUX_CMD : 2; ///< Bits 3-4: Comando SMUX (0: Inicialização do código ROM, 1: Ler configuração SMUX, 2: Escrever configuração SMUX)
    uint8_t : 3;          ///< Bits 5-7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_cfg6_t;

//CFG9 Register (Address 0xB2)
typedef union{
  struct{
    uint8_t : 4;           ///< Bits 0-3: Reservado
    uint8_t SIEN_SMUX : 1; ///< Bit 4: Interrupção do Sistema para Operação SMUX (0: Desabilitado, 1: Habilitado)
    uint8_t : 1;           ///< Bits 5: Reservado
    uint8_t SIEN_FD : 1;   ///< Bit 6: Interrupção do Sistema para Detecção de Cintilação (0: Desabilitado, 1: Habilitado)
    uint8_t : 1;           ///< Bit 7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_cfg9_t;

//PERS Register (Address 0xBD)
typedef union{
  struct{
    uint8_t APERS : 4; ///< Bits 0-3: Persistência da Interrupção Espectral (0-15)
    uint8_t : 4;       ///< Bits 4-7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_pers_t;



//********************** 10.2.9 FIFO Buffer Data and Status **********************
//FIFO_MAP  Register (Address 0xFC)
typedef union{
  struct{
    uint8_t FIFO_WRITE_ASTATUS : 1;  ///< Bit 0: Escrita de Status ASTATUS no FIFO (0: Desabilitado, 1: Habilitado)
    uint8_t FIFO_WRITE_CH0_DATA : 1; ///< Bit 1: Escrita de Dados CH0 no FIFO (0: Desabilitado, 1: Habilitado)
    uint8_t FIFO_WRITE_CH1_DATA : 1; ///< Bit 2: Escrita de Dados CH1 no FIFO (0: Desabilitado, 1: Habilitado)
    uint8_t FIFO_WRITE_CH2_DATA : 1; ///< Bit 3: Escrita de Dados CH2 no FIFO (0: Desabilitado, 1: Habilitado)
    uint8_t FIFO_WRITE_CH3_DATA : 1; ///< Bit 4: Escrita de Dados CH3 no FIFO (0: Desabilitado, 1: Habilitado)
    uint8_t FIFO_WRITE_CH4_DATA : 1; ///< Bit 5: Escrita de Dados CH4 no FIFO (0: Desabilitado, 1: Habilitado)
    uint8_t FIFO_WRITE_CH5_DATA : 1; ///< Bit 6: Escrita de Dados CH5 no FIFO (0: Desabilitado, 1: Habilitado)
    uint8_t : 1;                     ///< Bit 7: Reservado
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_fifo_map_t;

//FIFO_CFG0 Register (Address 0xD7)
typedef union{
  struct{
    uint8_t : 7;               ///< Bits 1-7: Reservado (0100001)
    uint8_t FIFO_WRITE_FD : 1; ///< Bit 0: Escrita de Detecção de Cintilação no FIFO (0: Desabilitado, 1: Habilitado)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_fifo_cfg0_t;

//FIFO_LVL  Register (Address 0xFD)
typedef union{
  struct{
    uint8_t FIFO_LVL : 8; ///< Bits 0-7: Nível do Buffer FIFO (0-128)
  };
  uint8_t value; ///< Valor bruto do registrador
} as7341_fifo_lvl_t;

//FDATA     Register (Address 0xFE and 0xFF)
typedef union{
  struct{
    uint8_t FDATA_L : 8; ///< Bits 0-7: Dados do Buffer FIFO - Byte inferior
    uint8_t FDATA_H : 8; ///< Bits 8-15: Dados do Buffer FIFO - Byte superior
  };
  uint16_t value; ///< Valor de 16 bits dos dados do buffer FIFO
} as7341_fdata_t;



//*************** SMUX I2C MAP *****************
typedef union{
  struct{
    uint8_t : 4;          ///< Bits 0-3: Unused
    uint8_t MUX_SEL : 3; ///< Bits 4-6:  Define Conexao aos ADC
    uint8_t : 1;          ///< Bit 7: Unused
  };
  uint8_t value; ///< 
} as7341_reg0;

typedef union{
  struct{
    uint8_t MUX_SEL: 3;          ///< Bits 0-3: Unused
    uint8_t : 5; 
  };
  uint8_t value; ///< 
} as7341_reg1;
typedef union{
  struct{
    uint8_t : 4;          ///< Bits 0-3: Unused
    uint8_t MUX_SEL : 3; ///< Bits 4-6:  Define Conexao aos ADC
    uint8_t : 1;          ///< Bit 7: Unused
  };
  uint8_t value; ///< 
} as7341_reg3;
typedef union{
  struct{
    uint8_t MUX_SEL: 3;          ///< Bits 0-3: Unused
    uint8_t : 5; 
  };
  uint8_t value; ///< 
} as7341_reg4;
typedef union{
  struct{
    uint8_t MUX_SEL1: 3;          ///< Bits 0-2: Pixel ID 10
    uint8_t : 1; 
    uint8_t MUX_SEL2: 3;          ///< Bits 4-6: Pixel ID 11
    uint8_t : 1; 
  };
  uint8_t value; ///< 
}as7341_reg5;
typedef union{
  struct{
    uint8_t : 4;          
    uint8_t MUX_SEL: 3; ///< Bits 4-6: Pixel ID 13
    uint8_t : 1; 
  };
  uint8_t value; ///< 
}as7341_reg6;
typedef union{
  struct{
    uint8_t MUX_SEL: 3;          ///< Bits 0-3: Unused
    uint8_t : 5; 
  };
  uint8_t value; ///< 
} as7341_reg7;
typedef union{
  struct{
    uint8_t : 4; 
    uint8_t MUX_SEL: 3;          ///< Bits 4-6: Pixel ID 11
    uint8_t : 1; 
  };
  uint8_t value; ///< 
}as7341_reg8;
typedef union{
  struct{
    uint8_t : 4; 
    uint8_t MUX_SEL: 3;          ///< Bits 4-6: Pixel ID 11
    uint8_t : 1; 
  };
  uint8_t value; ///< 
}as7341_reg9;
typedef union{
  struct{
    uint8_t MUX_SEL: 3;          ///< Bits 0-2: Pixel ID 10
    uint8_t : 5; 
  };
  uint8_t value; ///< 
}as7341_regA;
typedef union{
  struct{
    uint8_t : 4; 
    uint8_t MUX_SEL: 3;          ///< Bits 4-6: Pixel ID 11
    uint8_t : 1; 
  };
  uint8_t value; ///< 
}as7341_regC;
typedef union{
  struct{
    uint8_t MUX_SEL: 3;          ///< Bits 0-2: Pixel ID 10
    uint8_t : 5; 
  };
  uint8_t value; ///< 
}as7341_regD;
typedef union{
  struct{
    uint8_t MUX_SEL1: 3;          ///< Bits 0-2: Pixel ID 10
    uint8_t : 1; 
    uint8_t MUX_SEL2: 3;          ///< Bits 0-2: Pixel ID 10
    uint8_t : 1; 
  };
  uint8_t value; ///< 
}as7341_regE;
typedef union{
  struct{
    uint8_t : 4; 
    uint8_t MUX_SEL: 3;          ///< Bits 0-3: Unused
    uint8_t : 1; 
  };
  uint8_t value; ///< 
} as7341_regF;
typedef union{
  struct{
    uint8_t MUX_SEL1: 3;          ///< Bits 0-2: Pixel ID 10
    uint8_t : 1; 
    uint8_t MUX_SEL2: 3;          ///< Bits 4-6: Pixel ID 11
    uint8_t : 1; 
  };
  uint8_t value; ///< 
}as7341_reg0x10;
typedef union{
  struct{
    uint8_t MUX_SEL1: 3;          ///< Bits 0-2: Pixel ID 10
    uint8_t : 1; 
    uint8_t MUX_SEL2: 3;          ///< Bits 4-6: Pixel ID 11
    uint8_t : 1; 
  };
  uint8_t value; ///< 
}as7341_reg0x11;
typedef union{
  struct{
    uint8_t : 4; 
    uint8_t MUX_SEL: 3;          ///< Bits 0-3: Unused
    uint8_t : 1; 
  };
  uint8_t value; ///< 
} as7341_reg0x12;
typedef union{
  struct{
    uint8_t MUX_SEL1: 3;          ///< Bits 0-2: Pixel ID 10
    uint8_t : 1; 
    uint8_t MUX_SEL2: 3;          ///< Bits 4-6: Pixel ID 11
    uint8_t : 1; 
  };
  uint8_t value; ///< 
}as7341_reg0x13;

//-------------------------------------------
//-------------------------------------------
//-------------------------------------------
extern TaskHandle_t AS7341_Handle;
bool AS7341_pooling;

bool AS7341_SetAcessAndWrite(uint8_t regAdd, uint8_t data);
bool AS7341_SetAcessAndRead(uint8_t regAdd, uint8_t *data);
bool AS7341_write(uint8_t regAdd, uint8_t data);
bool AS7341_read(uint8_t regAdd, uint8_t *data);
bool AS7341_writeMultiples(uint8_t startReg, uint8_t *data, uint32_t length );
bool AS7341_readMultiples(uint8_t startReg, uint8_t *data, uint32_t length);

bool AS7341_i2cInit();
bool AS7341_Boot();

bool AS7341_Enable();
bool AS7341_DisableSpecMen();
bool AS7341_EnableSpecMen();
bool AS7341_PowerOff();
bool AS7341_PowerOn();
bool AS7341_DevivceConfig();
bool AS7341_WriteI2cReg2SMUX_Sel();

bool AS7341_ADC_TimingConfig();     
bool AS7341_ADC_Config();           
bool AS7341_InterruptionConfig();     
bool AS7341_DeviceStatus(uint8_t regAdd, uint8_t* data);
bool AS7341_SpecData(uint8_t regAdd, uint8_t* data);
bool AS7341_SpecStatus();
bool AS7341_OtherConfig();
bool AS7341_BufferData(uint8_t regAdd, uint8_t* data);
bool AS7341_BufferConfig();

bool AS7341_BankAcessSet(uint8_t RedAdd);
bool AS7341_SetSMUX(uint8_t* photoDiode, uint8_t* ADC_ID);
bool AS7341_SMUXEnable();
bool AS7341_SetI2cRegSMUX(uint8_t* photoDiode, uint8_t* ADC_ID);
bool AS7341_SetStepADC(uint16_t value);
uint16_t AS7341_GetStepADC();
bool AS7341_SetTimeADC(uint8_t value);
uint16_t AS7341_GetTimeADC();
float AS7341_GetIntegrationTimeADC();
bool AS7341_SetWtimeADC(uint8_t value);
bool AS7341_SetGainADC(uint8_t value);
uint16_t AS7341_GetGainADC();
bool AS7341_GetStatus(uint8_t *value);

bool AS7341_GetSMUX();
bool AS7341_ReadChannels(uint8_t* photoDiode, uint8_t* ADC_ID, uint8_t* ADC_count);

bool AS7341_SetSMUXMini(uint8_t* photoDiode, uint8_t* ADC_ID);
bool AS7341_ReadChannelsMini(uint8_t* photoDiode, uint8_t* ADC_ID, uint8_t* ADC_count);

bool AS7341_PerformanceDbgInit();
bool AS7341_PerformanceDbgSet();
bool AS7341_PerformanceDbgClr();
bool AS7341_AnalogAproxConfig(uint32_t freq);
bool AS7341_AnalogAproxDutySet(float PhotoValue);
bool AS7341_WaitIntSig();
void AS7341_Debaunce(void *ptr);
void AS7341_IntWatchDog(void *ptr);
void DelayUs(uint32_t us, TaskHandle_t *RefTask);

//void SetPhotoArray(uint8_t* photoDiode, uint8_t len, uint8_t*PhotoArray, uint8_t* ADC_config);


/*
    ----------------------------------------------
              Convertion Matrix
    ----------------------------------------------

  fatores para reconstrução espectral 380-100nm
  380  = index = 0
  1000 = index = 619

  F1         F2       F3         F4        F5         F6      F7         F8        CLEAR     NIR

*/
  // 850, 14

extern const float GeneralSpectralCorrectionMatrix[];
extern const float as7341_array1[];
#endif // AS7341_H

