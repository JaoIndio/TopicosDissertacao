
/*
  Diagrama Esquematico disponivel em:
  https://docs.google.com/drawings/d/1gcq39PQBJHOZxjQQRkpTzGgM4fl85WrFNZME5gocTac/edit
*/
#include "ADC_DMA.h"

#include "FreeRTOS.h"
#include "task.h"
#include "event_groups.h"

void GPIOFIntHandler(void) {
  // Clear the GPIO interrupt flag
  GPIOIntClear(GPIO_PORTF_BASE, GPIO_INT_PIN_4);

  // Trigger ADC Sequencer 3
  ADCProcessorTrigger(ADC0_BASE, 3);
}

void DMAIntHandler(void){
  // Clear the interrupt flag
  uDMAIntClear(UDMA_CHANNEL_ADC0);

  // Signal FreeRTOS event group

  UARTprintf("\r\t\t\t[ADC DMA] Full FIFO");
  xEventGroupSetBitsFromISR(BurstEventGroup, BURST_FIFO_FULL, NULL);
}
void InitGPIOTrigger(){
  // Enable GPIO port F
  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);

  // Configure PF4 as input with pull-up
  GPIOPinTypeGPIOInput(GPIO_PORTF_BASE, GPIO_PIN_4);
  GPIOPadConfigSet(GPIO_PORTF_BASE, GPIO_PIN_4, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);

  // Configure PF4 to detect falling edge
  GPIOIntTypeSet(GPIO_PORTF_BASE, GPIO_PIN_4, GPIO_FALLING_EDGE);

  GPIOIntEnable(GPIO_PORTF_BASE, GPIO_PIN_4); // Step 3: Configure the interrupt
  IntRegister(INT_GPIOF, GPIOFIntHandler);
  IntEnable(INT_GPIOF);
}
void InitADC(){
  // Enable ADC0 and GPIO port E
  SysCtlPeripheralEnable(SYSCTL_PERIPH_ADC0);
  //SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);

  // Configure PE3 as an ADC input
  GPIOPinTypeADC(GPIO_PORTE_BASE, GPIO_PIN_1);

  // Configure ADC0 sequencer 3 to trigger on an external signal (GPIO trigger)
  ADCSequenceConfigure(ADC0_BASE, 3, ADC_TRIGGER_EXTERNAL, 0);

  // Configure the step to sample AIN0 (PE3), enable interrupt, and mark end of sequence
  ADCSequenceStepConfigure(ADC0_BASE, 3, 0, ADC_CTL_CH0 | ADC_CTL_IE | ADC_CTL_END);

  // Enable DMA for ADC0
  ADCSequenceDMAEnable(ADC0_BASE, 3);

  // Enable the ADC sequencer
  ADCSequenceEnable(ADC0_BASE, 3);

}
void InitDMA(){
  // Enable the uDMA module
  SysCtlPeripheralEnable(SYSCTL_PERIPH_UDMA);

  // Enable the uDMA controller
  uDMAEnable();

  // Set the control table base (must be 1024-byte aligned)
  static uint8_t controlTable[1024] __attribute__((aligned(1024)));
  uDMAControlBaseSet(controlTable);

  // Configure DMA channel for ADC0
  uDMAChannelAttributeDisable(UDMA_CHANNEL_ADC0, UDMA_ATTR_ALL);
  uDMAChannelControlSet(UDMA_CHANNEL_ADC0 | UDMA_PRI_SELECT, \
                        UDMA_SIZE_32 | UDMA_SRC_INC_NONE | UDMA_DST_INC_32 | UDMA_ARB_1);
  uDMAChannelTransferSet(UDMA_CHANNEL_ADC0 | UDMA_PRI_SELECT, \
                         UDMA_MODE_BASIC,\
                         (void *)(ADC0_BASE + ADC_O_SSFIFO3), adcBuffer, ADC_BUFFER_SIZE);

  // Enable the DMA channel
  uDMAChannelEnable(UDMA_CHANNEL_ADC0);
}
void InitInterruptions(){
  // Enable DMA interrupts
  IntEnable(INT_UDMAERR);
  //IntEnable(INT_ADC0SS3);

  // Set DMA interrupt handler
  IntRegister(INT_UDMAERR, DMAIntHandler);
  IntPrioritySet(INT_UDMAERR, 0); // Set highest priority
}
bool BurstModeConfig(){
  BurstEventGroup = xEventGroupCreate();

  //InitGPIOTrigger();
  InitADC();
  InitDMA();
  InitInterruptions();

  return true;
}
