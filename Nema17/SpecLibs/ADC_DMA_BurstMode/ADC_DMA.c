
/*
  Diagrama Esquematico disponivel em:
  https://docs.google.com/drawings/d/1gcq39PQBJHOZxjQQRkpTzGgM4fl85WrFNZME5gocTac/edit
*/

/******************************
//! \note The interrupt handler for the uDMA is for transfer completion when
//! the channel UDMA_CHANNEL_SW is used and for error interrupts.  The
//! interrupts for each peripheral channel are handled through the individual
//! peripheral interrupt handlers.
****************************/

#include "ADC_DMA.h"

#include "FreeRTOS.h"
#include "task.h"
#include "event_groups.h"
#include "inc/hw_types.h"

#include "SpecResult/SpecResult.h"
#include "DRV8825/drv8825.h"
#include "MonoLight/mono_light.h"

#define ADC_SEQ 3
#define DMA_CHANNEL UDMA_CH17_ADC0_3
#define ADC_INT_SEQ INT_ADC0SS3


uint32_t adc_count =0;
uint32_t dma_count =0;
uint32_t* ADC_result = ADC0_BASE + ADC_O_SSFIFO3;
uint32_t* UART_Tx = UART5_BASE + UART_O_DR;
volatile uint8_t ADC_rslt[2];

void GPIOFIntHandler(void) {
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  uint32_t status = GPIOIntStatus(GPIO_PORTF_BASE, true);

  //char actualTask[] = "\t\t\t\t[GPIOF Handler]\t\t";
  // Clear the GPIO interrupt flag
  if(status & GPIO_PIN_4 ){ 
    GPIOIntClear(GPIO_PORTF_BASE, GPIO_INT_PIN_4);
    ADCTriggerDbgSet();
    // Trigger ADC Sequencer 3
    ADCProcessorTrigger(ADC0_BASE, ADC_SEQ);
    //xEventGroupSetBitsFromISR(BurstEventGroup, BURST_FIFO_FULL, NULL);
    //UARTprintf("%s ADC Trigger\n", actualTask);
  }else if(status & PWM_INT){
    GPIOIntClear(GPIO_PORTF_BASE, PWM_INT);
    if(!StepCount.Began) StepCount.Began=true;
  
    if(StepCount.CycleCount<StepCount.CycleThrshld)
      StepCount.Count++;

    if(StepCount.Count>=160*5){
    //if(StepCount.Count>=80){
      StepCount.Count = 0;
      StepCount.CycleCount++;
      vTaskNotifyGiveFromISR(xTrackInterfMovHandle, &xHigherPriorityTaskWoken);
    }
    if(StepCount.CycleCount>=StepCount.CycleThrshld){
      GreenLightTurnOff();
      GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, 0);
      NemaDisable();
      vTaskDelete(xTrackInterfMovHandle);
    }
  }
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

/******************************
// FONTE : ufsm/Mestrado/Dissertacao/Pototip_DeVereda/ProjetoPrototipo/Tiva/TM4C/driverlib/udma.c
//! \note The interrupt handler for the uDMA is for transfer completion when
//! the channel UDMA_CHANNEL_SW is used and for error interrupts.  The
//! interrupts for each peripheral channel are handled through the individual
//! peripheral interrupt handlers.
/RT0_BASE + UART_O_DR)
****************************/

uint32_t errorStatus; // = uDMAErrorStatusGet();
void ADCIntHanlder(void){
  char actualTask[] = "\t\t\t\t[ADC Handler]\t\t";
  adc_count++;
  ADCIntClear(ADC0_BASE, ADC_SEQ);
  //UARTprintf("%s ADC Done\n", actualTask);
//*
  //uint32_t ui32Mode = uDMAChannelModeGet(UDMA_CH7_UART5TX);
  //if(ui32Mode == UDMA_MODE_STOP){
////
    UART5_DbgSet();
    ADC_rslt[0] = (*((uint16_t*)(ADC0_BASE + ADC_O_SSFIFO3))>>8) & 0xFF; // MSB
    ADC_rslt[1] = (*((uint16_t*)(ADC0_BASE + ADC_O_SSFIFO3))) & 0xFF;    // LSB
    //ADC_rslt = *((uint16_t*)(ADC0_BASE + ADC_O_SSFIFO3));
    uDMAChannelTransferSet(UDMA_CH7_UART5TX | UDMA_PRI_SELECT, \
                           UDMA_MODE_BASIC,\
                           (void *) ADC_rslt, \
                           (void *)(UART5_BASE + UART_O_DR),\
                           2);

    uDMAChannelEnable(UDMA_CH7_UART5TX);
    errorStatus = uDMAErrorStatusGet();
/*
    uDMAChannelTransferSet(DMA_CHANNEL | UDMA_PRI_SELECT, \
                           UDMA_MODE_BASIC,\
                           (void *)(ADC0_BASE + ADC_O_SSFIFO0), adcBuffer+2, ADC_BUFFER_SIZE);
    uint32_t errorStatus = uDMAErrorStatusGet();
    uDMAChannelEnable(DMA_CHANNEL);
*/
    if(adc_count%200==0){
      //UARTprintf("\r\t\t\tADC count: %d\n", adc_count); // Example processing
    }
    if(adc_count==1024*4){
      //UARTprintf("\r\n\t\t\t\tDONE\n", adc_count); // Example processing
      xEventGroupSetBitsFromISR(BurstEventGroup, BURST_FIFO_FULL, NULL);
      //UARTprintf("\r\n\n"); // Example processing
      adc_count=0;
    }
    ADCTriggerDbgRst();
  //}
  taskYIELD();


}
void uDMAErrIntHandler(){
  char actualTask[] = "\t\t\t\t[DMA ERR Handler]\t\t";
  UARTprintf("%s [*] Here [*]\n", actualTask);
  BurstDMA_Check();
}
void uDMAIntHandler(void){
  char actualTask[] = "\t\t\t\t[DMA Handler]\t\t";
  dma_count++;
  // Clear the interrupt flag
  uDMAIntClear(DMA_CHANNEL);
  uDMAIntClear(INT_UDMA);

  // Signal FreeRTOS event group
  //UARTprintf("%s Full FIFO\n", actualTask);
  //xEventGroupSetBitsFromISR(BurstEventGroup, BURST_FIFO_FULL, NULL);
}
void InitGPIOTrigger(){
  char actualTask[] = "\t\t\t\t\t[Init GPIOF]\t\t";
  // Enable GPIO port F
  UARTprintf("%s Periph Enable\n", actualTask);
  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);

  // Configure PF4 as input with pull-up
  UARTprintf("%s Input Config\n", actualTask);
  GPIOPinTypeGPIOInput(GPIO_PORTF_BASE, GPIO_PIN_4);
  GPIOPadConfigSet(GPIO_PORTF_BASE, GPIO_PIN_4, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);

  // Configure PF4 to detect falling edge
  UARTprintf("%s INT Config\n", actualTask);
  GPIOIntTypeSet(GPIO_PORTF_BASE, GPIO_PIN_4, GPIO_RISING_EDGE);

  UARTprintf("%s INT Enable\n", actualTask);
  GPIOIntEnable(GPIO_PORTF_BASE, GPIO_PIN_4); // Step 3: Configure the interrupt
  UARTprintf("%s INT CallBackSet\n", actualTask);
  IntRegister(INT_GPIOF, GPIOFIntHandler);
  IntPrioritySet(INT_GPIOF, 0x6);
  IntEnable(INT_GPIOF);
  
  //Performance Pin Configuration
/*******************************************/
  GPIOUnlockPin(ADC_MONITOR_BASE, ADC_MONITOR_GPIO);
  GPIOPinTypeGPIOOutput(ADC_MONITOR_BASE, ADC_MONITOR_GPIO);
  GPIOPadConfigSet(ADC_MONITOR_BASE,  ADC_MONITOR_GPIO, \
                   GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD);
/*******************************************/

}
void InitADC(){
  char actualTask[] = "\t\t\t\t\t[InitADC]\t\t";
  UARTprintf("%s Periph Enable\n", actualTask);
  // Enable ADC0 and GPIO port E
  SysCtlPeripheralEnable(SYSCTL_PERIPH_ADC0);
  IntDisable(ADC_INT_SEQ);
  ADCIntDisable(ADC0_BASE, ADC_SEQ);
  ADCSequenceDisable(ADC0_BASE, ADC_SEQ);
  //SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE); U
  //UDMA_CH24_ADC1_0
  // Configure PE3 as an ADC input
  GPIOPinTypeADC(GPIO_PORTD_BASE, GPIO_PIN_2);
  UARTprintf("%s Config\n", actualTask);
  // Configure ADC0 sequencer 3 to trigger on an external signal (GPIO trigger)
  ADCSequenceConfigure(ADC0_BASE, ADC_SEQ, ADC_TRIGGER_PROCESSOR, 0);
  // Configure the step to sample AIN0 (PE3), enable interrupt, and mark end of sequence
  ADCSequenceStepConfigure(ADC0_BASE, ADC_SEQ, 0, ADC_CTL_CH5 | ADC_CTL_IE | ADC_CTL_END);

  // Enable the ADC sequencer
  UARTprintf("%s ADC Enable\n", actualTask);
  ADCSequenceEnable(ADC0_BASE, ADC_SEQ);
  ADCIntClear(ADC0_BASE, ADC_SEQ);
  ADCIntEnable(ADC0_BASE, ADC_SEQ);
  IntPrioritySet(ADC_INT_SEQ, 0x1); // Set highest priority
  
  //UARTprintf("%s DMA Link\n", actualTask);
  // Configure ADC0 sequencer 3 to trigger on an external signal (GPIO trigger)
  // Enable DMA for ADC0
  ADCSequenceDMAEnable(ADC0_BASE, ADC_SEQ);
}
void InitDMA(){
  // Enable the uDMA module
  //SysCtlPeripheralEnable(SYSCTL_PERIPH_UDMA);

  // Enable the uDMA controller
  //uDMAEnable();

  // Set the control table base (must be 1024-byte aligned)
  //static uint8_t controlTable[1024] __attribute__((aligned(1024)));
  //uDMAControlBaseSet(pui8ControlTable);

  // Enable DMA interrupts
  IntPrioritySet(INT_UDMA, 0x0); // Set highest priority
  IntEnable(INT_UDMA);
  IntEnable(INT_UDMAERR);
  // Set DMA interrupt handler
  uDMAIntRegister(INT_UDMA,    uDMAIntHandler);
  //uDMAIntRegister(INT_UDMAERR, uDMAErrIntHandler);
  
  uDMAChannelAssign(DMA_CHANNEL);

  // Configure DMA channel for ADC0
  uDMAChannelAttributeDisable(DMA_CHANNEL, 
                              UDMA_ATTR_ALTSELECT | UDMA_ATTR_HIGH_PRIORITY|
                              UDMA_ATTR_REQMASK);
  
  uDMAChannelControlSet(DMA_CHANNEL | UDMA_PRI_SELECT, \
                        UDMA_SIZE_16 | UDMA_SRC_INC_NONE | UDMA_DST_INC_16 | UDMA_ARB_8);
/// *
  uDMAChannelTransferSet(DMA_CHANNEL | UDMA_PRI_SELECT, \
                         UDMA_MODE_BASIC,\
                         (void *)(ADC0_BASE + ADC_O_SSFIFO3), adcBuffer, ADC_BUFFER_SIZE);
//*/
  uDMAChannelAttributeEnable(DMA_CHANNEL, UDMA_ATTR_HIGH_PRIORITY);
  uDMAChannelAttributeEnable(DMA_CHANNEL, UDMA_ATTR_USEBURST);

  // Enable the DMA channel
  uDMAChannelEnable(DMA_CHANNEL);
  //uDMAChannelRequest(DMA_CHANNEL); DMA_CHANNEL
}
void InitInterruptions(){
  IntPrioritySet(ADC_INT_SEQ, 0x7); // Set highest priority
  IntEnable(ADC_INT_SEQ);
  ADCIntEnableEx(ADC0_BASE, ADC_INT_SS0|ADC_INT_DMA_SS0);
  IntRegister(ADC_INT_SEQ, ADCIntHanlder);
}
bool BurstModeConfig(){
  char actualTask[] = "\t\t\t\t[Burst Cfg]\t\t";
  BurstEventGroup = xEventGroupCreate();
  
  adcBuffer[0] = 0xAA;
  adcBuffer[1] = 0xAA;
  adcBuffer[ADC_BUFFER_SIZE+4-2] = 0x55;
  adcBuffer[ADC_BUFFER_SIZE+4-1] = 0x55;

  UARTprintf("%s Init DMA\n", actualTask);
  InitDMA();
  UARTprintf("%s Init ADC\n", actualTask);
  InitADC();
  UARTprintf("%s Init Interruption\n", actualTask);
  InitInterruptions();
  
  UARTprintf("%s Init GPIOF Trigger\n", actualTask);
  InitGPIOTrigger();

  return true;
}

void BurstDMA_Check(){
  if (uDMAChannelIsEnabled(DMA_CHANNEL)) {
    UARTprintf("DMA channel is enabled\n");
  }else {
    UARTprintf("DMA channel is NOT enabled\n");
  }
  
  if (HWREG(ADC0_BASE + ADC_O_ACTSS) & ADC_ACTSS_ASEN3) {
    UARTprintf("ADC Sequencer 3 is enabled\n");
    if (HWREG(ADC0_BASE + ADC_O_SSDC3) & ADC_ACTSS_ADEN3) {
      UARTprintf("ADC DMA Request is enabled\n");
    }else {
      UARTprintf("ADC DMA Request is NOT enabled\n");
    }
  }
  uint32_t mode = uDMAChannelModeGet(DMA_CHANNEL | UDMA_PRI_SELECT);
  if (mode == UDMA_MODE_STOP) {
    UARTprintf("DMA transfer is complete or not started\n");
  }else if (mode == UDMA_MODE_BASIC) {
    UARTprintf("DMA is in BASIC mode and active\n");
  }else {
    UARTprintf("DMA is in unknown mode\n");
  }
  uint32_t status = uDMAErrorStatusGet();
  if (status) {
    UARTprintf("DMA Error Status: 0x%08x\n", status);
    uDMAErrorStatusClear(); // Clear DMA error status
  }else {
    UARTprintf("No DMA errors detected 0x%08x\n", status);
  }
  if (((uint32_t)pui8ControlTable & 0x3FF) == 0) {
    UARTprintf("Control table alignment OK\n");
  } else {
    UARTprintf("Control table is NOT aligned!\n");
  }
}
void ADCTriggerDbgRst(){
  GPIOPinWrite(ADC_MONITOR_BASE, ADC_MONITOR_GPIO, 0);
}
void ADCTriggerDbgSet(){
  GPIOPinWrite(ADC_MONITOR_BASE, ADC_MONITOR_GPIO, ADC_MONITOR_GPIO);
}
/******************************
//! \note The interrupt handler for the uDMA is for transfer completion when
//! the channel UDMA_CHANNEL_SW is used and for error interrupts.  The
//! interrupts for each peripheral channel are handled through the individual
//! peripheral interrupt handlers.
//
******************************/

  
