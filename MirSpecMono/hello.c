#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

/* Kernel includes. */
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

/* Hardware includes. */
#include "inc/hw_ints.h" //Mapeamento de interrupcoes
#include "inc/hw_memmap.h"
#include "inc/hw_sysctl.h"
#include "driverlib/gpio.h"
#include "driverlib/interrupt.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "driverlib/uart.h"
#include "drivers/rtos_hw_drivers.h"
#include "utils/uartstdio.h"

/*  -----  My Libs  ----  */
#include "DRV8825/drv8825.h"
#include "AS7341/AS7341.h"
#include "SpecResult/SpecResult.h"
#include "LinearMov/LinMov.h"
#include "ADC_DMA_BurstMode/ADC_DMA.h"
/*  -----  My Libs  ----  */

#include "arm_math.h"

//#include "myLib.h"
//#include "external_devices/AS7341_photo.h"
#define SPEC_TOTAL_WAVELENGHT 620
volatile float IntTime;
volatile float Gain;
volatile float32_t Spectral400_600[SPEC_TOTAL_WAVELENGHT];

/**
 * hello.c
 */
/* Set up the hardware ready to run this demo. */
static void prvSetupHardware( void );

/* This function sets up UART0 to be used for a console to display information
 * as the example is running. */
static void prvConfigureUART(void);
void joinADC();

void ADC_DMA_Reader();

volatile float PhotoOffset[11]={ 0.003347963f,0.005573356f,\
                                 0.007078014f,0.008031754f,\
                                 0.009154323f,0.009568005f,\
                                 0.011294808f,0.015861109f};

volatile float GainCr[11]={1.057724938f,1.04913698f,\
                           1.047883152f,1.049114772f,\
                           1.020740724f,1.015773647f,\
                           1.010904254f,1.0f,1.000322557f,\
                           0.987308373f,0.959349244f};

extern const float as7341_array1[3];

/// *
void SpectralReconstruction(float* PhotoCorrection){
/*
https://www.ti.com/lit/an/spma041g/spma041g.pdf
*/
  //float a = as7341_array1[1];
  arm_matrix_instance_f32 matA; 
  arm_matrix_instance_f32 matB; 
  arm_matrix_instance_f32 matC;

  arm_mat_init_f32(&matA, SPEC_TOTAL_WAVELENGHT, 10, (float32_t*)GeneralSpectralCorrectionMatrix);
  arm_mat_init_f32(&matB, 10,  1,  (float32_t*)PhotoCorrection);
  arm_mat_init_f32(&matC, SPEC_TOTAL_WAVELENGHT,  1, (float32_t*)Spectral400_600);
  
  arm_status multResult;
  multResult = arm_mat_mult_f32(&matA, &matB, &matC);
  
  if(multResult!=ARM_MATH_SUCCESS)
    UARTprintf("\r\t\t[Spec Rec] ERROR %d\n", multResult);
}
//*/
void BasicCountConvertion(float*PhotoCr ){
  uint8_t index =0;
  while(index<10){
    //Basic count = Raw/(GAIN*IntegrationTime)
    //PhotoCr[index] = PhotoCr[index]/64.0f*IntTime;
    PhotoCr[index] = PhotoCr[index]/Gain*IntTime;
    if(index<8)
      PhotoCr[index] +=PhotoOffset[index];
    index++;
  }
}

void Correction1(uint16_t* ADC_Raw, bool round, float*PhotoCr ){
  uint8_t index =0;
  uint8_t raw_index =0;
  if(round){
    while(index<6){
      PhotoCr[index] = ADC_Raw[index];
      index++;
    }
  }else{
    index = 6;
    raw_index=0;
    while(index<10){
      if(raw_index!=3)
        PhotoCr[index] = ADC_Raw[raw_index];
      else{
        PhotoCr[index] = ADC_Raw[5];
        break;
      }
      raw_index++;
      index++;
    }
  }
}

void joinADC(uint8_t* ADC_count, uint16_t* ADC_raw){
  uint8_t i =0;
  uint8_t j =0;
  uint16_t data_aux1, data_aux2;
  for(i=0;i<12;i+=2){
    //UARTprintf("\rADC_count[%d]: %u | %u\n",j,ADC_count[i+1], ADC_count[i]);
    data_aux1 = (uint16_t)ADC_count[i];
    data_aux2 = (uint16_t)ADC_count[i+1];
    ADC_raw[j] = data_aux1 | data_aux2<<8;
    //UARTprintf("\rADC_raw[%d]: %u\n",j,ADC_raw[j]);
    j++;
  }
}
void AS7341_Begin(void *ptr){
  
  NemaInterruptionConfig();
  if(!AS7341_Boot())
    UARTprintf("AS7341 Boot ERROR\n");
  else
    UARTprintf("AS7341 Boot Done\n");
  uint8_t photoDiode[18];
  uint8_t ADC_ID[18];
  uint8_t ADC_ID2[18];
  bool round = false;
  uint8_t ADC_count[12];
  uint16_t ADC_raw[6];
  float PhotoCorrection[10];
  //float PhotoClearNir[10];
  
  uint8_t index;
  // Check stack usage periodically
  UBaseType_t unusedStackWords = uxTaskGetStackHighWaterMark(NULL);
  size_t unusedStackBytes = unusedStackWords * sizeof(StackType_t);
  
  bool BurstResult;
    //BurstResult = BurstModeConfig();

  UARTprintf("\rUnused stack memory: %u bytes\n", (unsigned int)unusedStackBytes);
                          
  //for(index=0; index<12; index++) ADC_count[index] = 0x05;

  //uint8_t readCheck;
  // Definições especiais
  /*                      i2c Reg   |  IDs    |   PHOTO  |
                        ------------------------------------
                             0x5     11 e 10     F4_2 e F2_2              
                             0xE     29 e 28     F6_2 e F8_2             
                            0x10     33 e 32   GPIO e F1_2               
                            0x11     35 e 34     C2 e INT            
                            0x13     39 e 38   FLKR e NIR            
  */
  photoDiode[0] =  PHOTO_F1_1;
  photoDiode[1] =  PHOTO_F3_1;
  photoDiode[2] =  PHOTO_F5_1;
  photoDiode[3] =  PHOTO_F7_1;
  photoDiode[4] =  PHOTO_F6_1;
  photoDiode[5] =  PHOTO_F8_1;
  photoDiode[6] =  PHOTO_F2_1;
  photoDiode[7] =  PHOTO_F4_1;
  
  photoDiode[8]  =  PHOTO_F4_2<<4 | PHOTO_F2_2;
  photoDiode[9]  =  PHOTO_F6_2<<4 | PHOTO_F8_2;
  photoDiode[10] =  PHOTO_F7_2;
  photoDiode[11] =  PHOTO_F5_2;
  photoDiode[12] =  PHOTO_F3_2;
  photoDiode[13] =  GPIO_INPUT<<4 | PHOTO_F1_2;
  photoDiode[14] =  PHOTO_CLEAR_1;
  photoDiode[15] =  PHOTO_CLEAR_2<<4 | INT_INPUT;
  photoDiode[16] =  PHOTO_FLICKER<<4 | PHOTO_NIR;
  
  photoDiode[17] =  DARK;
  
  ADC_ID[0] =  CONNECT_TO_ADC0; 
  ADC_ID[1] =  CONNECT_TO_ADC2;    
  ADC_ID[2] =  CONNECT_TO_ADC4;    
  ADC_ID[3] =  CONNECT_TO_GND;    
  ADC_ID[4] =  CONNECT_TO_ADC5;    
  ADC_ID[5] =  CONNECT_TO_GND;    
  ADC_ID[6] =  CONNECT_TO_ADC1;    
  ADC_ID[7] =  CONNECT_TO_ADC3;   
  
  ADC_ID[8]  = CONNECT_TO_ADC3<<4 | CONNECT_TO_ADC1;    
  ADC_ID[9]  = CONNECT_TO_ADC5<<4 | CONNECT_TO_GND;   
  ADC_ID[10] = CONNECT_TO_GND;   
  ADC_ID[11] = CONNECT_TO_ADC4;   
  ADC_ID[12] = CONNECT_TO_ADC2;   
  ADC_ID[13] = CONNECT_TO_GND<<4 | CONNECT_TO_ADC0;   
  ADC_ID[14] = CONNECT_TO_GND;   
  ADC_ID[15] = CONNECT_TO_GND<<4 | CONNECT_TO_GND;   
  ADC_ID[16] = CONNECT_TO_GND    | CONNECT_TO_GND;   
  
  ADC_ID[17] = CONNECT_TO_GND;   

  ADC_ID2[0] =  CONNECT_TO_GND; 
  ADC_ID2[1] =  CONNECT_TO_GND;    
  ADC_ID2[2] =  CONNECT_TO_ADC0;    
  ADC_ID2[3] =  CONNECT_TO_GND;    
  ADC_ID2[4] =  CONNECT_TO_GND;    
  ADC_ID2[5] =  CONNECT_TO_GND;    
  ADC_ID2[6] =  CONNECT_TO_GND;    
  ADC_ID2[7] =  CONNECT_TO_ADC1;   
  
  ADC_ID2[8]  = CONNECT_TO_ADC1<<4 | CONNECT_TO_GND;    
  ADC_ID2[9]  = CONNECT_TO_GND<<4 | CONNECT_TO_GND;   
  ADC_ID2[10] = CONNECT_TO_GND;   
  ADC_ID2[11] = CONNECT_TO_ADC0;   
  ADC_ID2[12] = CONNECT_TO_GND;   
  ADC_ID2[13] = CONNECT_TO_GND<<4 | CONNECT_TO_GND;   
  ADC_ID2[14] = CONNECT_TO_GND;   
  ADC_ID2[15] = CONNECT_TO_GND<<4 | CONNECT_TO_GND;   
  ADC_ID2[16] = CONNECT_TO_GND    | CONNECT_TO_GND;   
  
  ADC_ID2[17] = CONNECT_TO_GND;
  
  //𝑡𝑖𝑛𝑡 = (𝐴𝑇𝐼𝑀𝐸 + 1) × (𝐴𝑆𝑇𝐸𝑃 + 1) × 2.78μ𝑠
  // 𝐴𝐷𝐶𝑓𝑢𝑙𝑙𝑠𝑐𝑎𝑙𝑒 = (𝐴𝑇𝐼𝑀𝐸 + 1) × (𝐴𝑆𝑇𝐸𝑃 + 1)
  // Step=1 e Time=1 resulta em uma leiutra e reconstrução completa em 70ms=+-14HZ
  uint16_t StepADC = 70;
  if(!AS7341_SetStepADC(StepADC))
    UARTprintf("\rSet STEP Error\n");
  uint8_t TimeADC  = 100;
  if(!AS7341_SetTimeADC(TimeADC))
    UARTprintf("\rSet Time Error\n");

  uint8_t Wtime_value;
  Wtime_value = (StepADC+1)*(TimeADC+1)*2.78f/1000.0f; // time in ms
  uint8_t wtime_value = (Wtime_value/2.78)+2;
  AS7341_SetWtimeADC(wtime_value);

  // ****Aumentar Tempo de Intetracao***
  if(!AS7341_SetGainADC(9))
    UARTprintf("\rSet GAIN Error\n");
  
  //Boot -> ReadChennels -> SetSMUX -> SetI2cRegSMUX
  uint16_t i;
  
  uint32_t ui32SysClock  = SysCtlClockGet();
  IntTime = AS7341_GetIntegrationTimeADC();
  Gain    = pow(2,(int)AS7341_GetGainADC()-1);

  UARTprintf("\rUART Init Gain %d\n", (int)Gain);
  UART5_Init(115200);
  UARTprintf("\rUART Init Done\n");
  as7341_status_t photo_status;
  as7341_status2_t photo_saturation;
  float ADC_fullscale = (float)((AS7341_GetStepADC()+1)*(AS7341_GetTimeADC()+1));
  float F5_intensity;

  AS7341_PerformanceDbgInit();
  as7341_stat_t stat_rslt;
  as7341_status2_t status2_rslt;
  as7341_status_t status_rslt;
  as7341_astatus_t astat2_rslt;
  if(!AS7341_SetSMUXMini(photoDiode, ADC_ID2))
    UARTprintf("\rSMUX Config Error\n");

  while(1){
    AS7341_WaitIntSig();
    AS7341_BankAcessSet(AS7341_REG_CH0_DATA_L);
    AS7341_read(AS7341_REG_CH0_DATA_L, ADC_count);
    AS7341_read(AS7341_REG_CH0_DATA_H, ADC_count+1);
    AS7341_read(AS7341_REG_CH1_DATA_L, ADC_count+2);
    AS7341_read(AS7341_REG_CH1_DATA_H, ADC_count+3);
    
    AS7341_DeviceStatus(AS7341_REG_STATUS,  &status_rslt.value);
    AS7341_SetAcessAndWrite(AS7341_REG_STATUS, status_rslt.value);
    
    joinADC(ADC_count, ADC_raw);
    F5_intensity = (float)ADC_raw[0]*1.8f/ADC_fullscale;
    UARTprintf("-------------------------------\n");
    UARTprintf("\t\t\t\tADC Full Scale %d\n", (int)(ADC_fullscale*10000));
    UARTprintf("\t\t\t\tADC Raw F5     %x\n", ADC_raw[0]);
    UARTprintf("\t\t\t\t* F5 Voltage   %d\n", (int)(F5_intensity*10000));
    UARTprintf("\n\n");
  }

  while(1){
    if(round){
/*
  BasicCount =       Raw_counts
                --------------------
                Gain*IntegrationTime
  AGAIN = 7 = 2⁶ = 64
  ATIME = 1 = ?  = ?
        Ordem das leituras
          F1, F2, F3, F4, F5, F6
*/
      AS7341_PerformanceDbgSet();    

      //if(!AS7341_ReadChannelsMini(photoDiode, ADC_ID, ADC_count))
      if(!AS7341_ReadChannels(photoDiode, ADC_ID, ADC_count))
        UARTprintf("\rErro de Leitura dos Canais\n");

      joinADC(ADC_count, ADC_raw);
      Correction1(ADC_raw, round, PhotoCorrection);
      round=false;
      AS7341_PerformanceDbgClr();

    }else{
/*
        Ordem das leituras
          F7, F8,CLEAR, F4, F5, NIR
*/    
      
      AS7341_PerformanceDbgSet();    
      //if(!AS7341_ReadChannelsMini(photoDiode, ADC_ID2, ADC_count))
      if(!AS7341_ReadChannels(photoDiode, ADC_ID2, ADC_count))
        UARTprintf("\rErro de Leitura dos Canais\n");
      joinADC(ADC_count, ADC_raw);
      Correction1(ADC_raw, round, PhotoCorrection);
      BasicCountConvertion(PhotoCorrection);

      //ADCProcessorTrigger(ADC0_BASE, 3);
      
      SpectralReconstruction(PhotoCorrection);
/*
  
*/
      //Eh preciso estimar a intensidade (em Volts) do fotodiodo em questão e comparar a reconstrucao espectral
      // baseada na estimativa com a matrix de reconstrucao
      // Reconstrucao [Tentativa 1] -> 1V8/ADCfullscale => 1V8*ADC_count[F5]/ADCfullscale
      //    Dps de reconstruir a intensidade luminosa eu preciso associar ele ao deslocamento do espelho.
      //    Armazena-se a variação da intensidade em função do deslocamento e ao fim
      //    Faz-se a FFT desse sinal.
      //    Compara-se a função espctral fruto da FFT com a adquirida pela reconstrução espectral
      //
      //      Expectativa: Gerar uma onda senoidal a partir do F5. Essa onda senoidal vai servir de referencia posteriormente 
      //        como frequencia amostral da luz infravermelha
      //  Referencias:
      //    [1] Fourier-Transform Spectroscopy Instrumentation Engineering (Cap. 3 "Principios de Operacao")
      //    [2] Dissertação MIT (Cap. 2) [/home/jao/curso/ufsm/Mestrado/Dissertacao/Pototip_DeVereda/DesignRefs/FTNIR/DissertacaoMIT.pdf]
/*
      É fundamental amostrar o sinal IR com precisão, repetibilidade e associação com a posição do espelho. 
      Esse requisito impacta na implementação da interferometria da luz. Para [2], há duas possibilidades.
      A primeira é o uso de amostragens regulares baseadas no controle de velocidade do movimento do motor
      O segundo faz uso do sinal de refereêcia monocromático. A vanttagem do segundo é que a onda senoidal esta
      diretamente relaciona à posição do espelho móvel ao mesmo tempo que pode ser usada como referência para a
      frequência amostral do detector IR.
*/
      //FFT_Spec();
      
      AS7341_PerformanceDbgClr();   
      //UART5_SendDataPacket(Photo4|Photo5, 32); //500-560
      UART5_SendDataPacket(Spectral400_600, SPEC_TOTAL_WAVELENGHT);
      round=true;
      vTaskDelay(pdMS_TO_TICKS(25));
    }
    F5_intensity = ADC_raw[4]*1.8f/ADC_fullscale;

    // Talvez nao seja necessario dar um delay, mas
    //  mas sim um YIELD()
    taskYIELD();
    //vTaskDelay(pdMS_TO_TICKS(25));

    //AS7341_PerformanceDbgSet();    
    AS7341_DeviceStatus(AS7341_REG_STATUS, &photo_status.value);
    if(photo_status.ASAT==1){
      //Houve Saturacao
      AS7341_DeviceStatus(AS7341_REG_STATUS2, &photo_saturation.value);
      //if(photo_saturation.ASAT_DIGITAL)
        //UARTprintf("\rTempo de Integracao muito longo\n");
      //else if(photo_saturation.ASAT_ANALOG)
        //UARTprintf("\rLuz Ambiente Muito Intensa, considere reduzir o GANHO\n");
    }

    //AS7341_PerformanceDbgClr();    

/*
    UARTprintf("\r-------------- %d -----------------\n", round);
    for(i=0;i<6;i++) 
      UARTprintf("\rADC_raw[%d]: %d\n",i, ADC_raw[i]);
    UARTprintf("\r\n\n");
*/  
  }
  //vTaskDelete(NULL);
}

int main(void){
  prvSetupHardware();
  NemaConfig();
  //">CCS App Center</a> to oinstall othe compiler of  the required version, or migrate the project to one of the available compiler versions by adjusting project properties. EQU_Firmware_L0 properties Proble
  //LinearMovValidation();
  //UARTprintf("Hello World!\n");
  
  //verificar se criou certo
  xTaskCreate(AS7341_Begin, "AS7341", configMINIMAL_STACK_SIZE+50, \
                NULL, 14, \
                NULL);
/*
  xTaskCreate(ADC_DMA_Reader, "AdcDMA", configMINIMAL_STACK_SIZE+50, \
                NULL, configMAX_PRIORITIES-2, \
                NULL);
*/

  vTaskStartScheduler();
  while(1){ 

  }

	//return 0;
}


static void prvConfigureUART(void)
{
    /* Enable GPIO port A which is used for UART0 pins.
     * TODO: change this to whichever GPIO port you are using. */
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);

    /* Configure the pin muxing for UART0 functions on port A0 and A1.
     * This step is not necessary if your part does not support pin muxing.
     * TODO: change this to select the port/pin you are using. */


    /* Enable UART0 so that we can configure the clock. */
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);
    MAP_GPIOPinConfigure(GPIO_PA0_U0RX);
    MAP_GPIOPinConfigure(GPIO_PA1_U0TX);
    /* Use the internal 16MHz oscillator as the UART clock source. */

    /* Select the alternate (UART) function for these pins.
     * TODO: change this to select the port/pin you are using. */
    MAP_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    UARTClockSourceSet(UART0_BASE, UART_CLOCK_PIOSC);


    /* Initialize the UART for console I/O. */
    UARTStdioConfig(0, 115200, 16000000);
}
/*-----------------------------------------------------------*/

static void prvSetupHardware( void )
{
    /* Run from the PLL at 80 MHz.  Any updates to the PLL rate here would
     * need to be reflected in FreeRTOSConfig.h by updating the value of
     * configCPU_CLOCK_HZ with the new system clock frequency. */
    MAP_SysCtlClockSet(SYSCTL_SYSDIV_4 | SYSCTL_USE_PLL | SYSCTL_OSC_MAIN |
                       SYSCTL_XTAL_16MHZ);

    /* Configure device pins. */
    PinoutSet(false); //HAbilita varios perifericos, incluindo todos as GPIOS

    /* Configure UART0 to send messages to terminal. */
    prvConfigureUART();

}

void vApplicationTickHook( void )
{
    /* This function will be called by each tick interrupt if
        configUSE_TICK_HOOK is set to 1 in FreeRTOSConfig.h.  User code can be
        added here, but the tick hook is called from an interrupt context, so
        code must not attempt to block, and only the interrupt safe FreeRTOS API
        functions can be used (those that end in FromISR()). */

}

void vApplicationIdleHook( void )
{
    /* vApplicationIdleHook() will only be called if configUSE_IDLE_HOOK is set
    to 1 in FreeRTOSConfig.h.  It will be called on each iteration of the idle
    task.  It is essential that code added to this hook function never attempts
    to block in any way (for example, call xQueueReceive() with a block time
    specified, or call vTaskDelay()).  If the application makes use of the
    vTaskDelete() API function (as this demo application does) then it is also
    important that vApplicationIdleHook() is permitted to return to its calling
    function, because it is the responsibility of the idle task to clean up
    memory allocated by the kernel to any task that has since been deleted. */
}
/*-----------------------------------------------------------*/

void vApplicationStackOverflowHook( TaskHandle_t pxTask, char *pcTaskName )
{
    ( void ) pcTaskName;
    ( void ) pxTask;

    /* Run time stack overflow checking is performed if
    configCHECK_FOR_STACK_OVERFLOW is defined to 1 or 2.  This hook
    function is called if a stack overflow is detected. */
    IntMasterDisable();
    for( ;; );
}

void ADC_DMA_Reader(){
  uint32_t adcValue;
  EventBits_t events;
  while(1){
    events = xEventGroupWaitBits(BurstEventGroup,\
                                 BURST_FIFO_FULL, \
                                 pdTRUE, \
                                 pdFALSE,\
                                 portMAX_DELAY);

    if (events & BURST_FIFO_FULL) {
      UARTprintf("\r\t\t\t[ADC DMA Reader]\n");
      // Process the ADC result
     // printf("ADC Value: %lu\n", adcValue); // Example processing
   }
  } 
}
