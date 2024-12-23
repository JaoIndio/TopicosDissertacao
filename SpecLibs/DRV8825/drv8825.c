#ifndef _DRV_LIB_
#define _DRV_LIB_

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
#include "inc/hw_types.h"
#include "driverlib/timer.h"
#include "DRV8825/drv8825.h"
#include "LinearMov/LinMov.h"
#include "MonoLight/mono_light.h"

// Pin definitions
#define STEP_PIN              GPIO_PIN_5 //PB5
#define DIR_PIN               GPIO_PIN_0  //PB0
#define ENABLE_PIN            GPIO_PIN_1
#define SLEEP_PIN             GPIO_PIN_2
#define ANALOG_SIMULATE       GPIO_PIN_7
#define LINEAR_MOV_DBG        GPIO_PIN_0
#define LINEAR_MOV_DBG_BASE   GPIO_PORTE_BASE

// Dps de comprar um novo DRV consegui modular a freq do PWM de 100 até 30KHz sem problemas
// no motor

#define PWM_FREQUENCY 1000// <- Unica freq q consegui até agr com microstep de 32
// DRV configurado no seu potenciometro de forma a limitar a corrente em 
// 90mA-100mA com um PWM de 20KHz(osciloscópio 20.56KHz) foi uma das performances mais
// estaveis observadas

#define KILO_HZ 1000 // 1 kHzI

// Define constants for the sigmoid function
#define SIGMOID_K 0.001
#define SIGMOID_X0 5000.0

#define EC_1 GPIO_PIN_2  //PB0
#define EC_2 GPIO_PIN_3
//PE2 PE3 Handlers
// Task handle
/*
        =========================================================
    ---------------------------------------------------------------------

                Nema17        200 steps/revolution
               DRV8825        1/32 Microstep (DRVstep)
      Fianl Steper Rev        6400 step/revolution
              Gt2 Belt        20 teeth
                              2mm/teeth
                              40mm/rev
             Final Mov        0.00625mm/step ->6.25µm/DRVstep

      Ex:
            VelFinal    5mm/s
        * A polia desloca-se 40mm por revolução

               6400*VelFinal
            ------------------- = 800 steps/s = PWM de 800 Hz
                    40

    ---------------------------------------------------------------------
        =========================================================
*/
TaskHandle_t xDebaunceKeyHandle     = NULL;
TaskHandle_t xChangeDirectionHandle = NULL;
LinMovCycle_t LinearMov_Mngr;

uint32_t ChangeDirStatus;

void Timer0IntHandler(){
  TimerIntClear(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
  GPIOPinWrite(LINEAR_MOV_DBG_BASE, LINEAR_MOV_DBG,\
               GPIOPinRead(LINEAR_MOV_DBG_BASE, LINEAR_MOV_DBG)^LINEAR_MOV_DBG);
  //NemaDisable();
}

// Function to calculate the sigmoid value
void     TriggerPWMSigmoidFrequency(float* actual_freq, float target_freq);
uint32_t getPWMFrequency();
float    sigmoid(float x);
void     xDebaunceKey(void *ptr);
void     xChangeDirection(void *ptr);
void     AnalogInit();
void     PWM_SetDutyCycle(float dutyCycle);

void PWM_SetDutyCycle(float dutyCycle){
  uint32_t load = PWMGenPeriodGet(PWM0_BASE, PWM_GEN_0);
  if (dutyCycle < 0.0f) dutyCycle = 0.0f;
  else if (dutyCycle > 100.0f) dutyCycle = 100.0f;
  uint32_t pulseWidth = (uint32_t)((load * dutyCycle) / 100.0f);

  //UARTprintf("\r\t\tDuty %d\n", (int)((load * pulseWidth) / 100) );
  PWMPulseWidthSet(PWM0_BASE, PWM_OUT_1, pulseWidth);
}
void AnalogInit(){
  //SysCtlPeripheralEnable(SYSCTL_PERIPH_PWM0);

  // Configure the pin muxing for PWM1 on PB7
  GPIOPinConfigure(GPIO_PB7_M0PWM1);
  
  // Configure the PWM generator for countdown mode with immediate updates to the parameters
  PWMGenConfigure(PWM0_BASE, PWM_GEN_0, PWM_GEN_MODE_DOWN | PWM_GEN_MODE_NO_SYNC);
  // Set the PWM period to 10 kHz
  uint32_t pwmClock = SysCtlClockGet() / 64; // Assuming a PWM clock divider of 64
  uint32_t load = (pwmClock / KILO_HZ) - 1; // For a 10 kHz frequency
  PWMGenPeriodSet(PWM0_BASE, PWM_GEN_0, load);

  // Set the PWM duty cycle to 50% initially
  PWMPulseWidthSet(PWM0_BASE, PWM_OUT_1, load / 2);

  // Enable the PWM output
  PWMOutputState(PWM0_BASE, PWM_OUT_1_BIT, true);

  // Enable the PWM generator
  PWMGenEnable(PWM0_BASE, PWM_GEN_0);
}

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-SIGMOID_K * (x - SIGMOID_X0)));
}

// Function to get the current PWM frequency
uint32_t getPWMFrequency() {
  uint32_t systemClock = SysCtlClockGet();
  uint32_t pwmClockDivider = SysCtlPWMClockGet();
  uint32_t pwmClock = systemClock / pwmClockDivider;
  uint32_t loadValue = PWMGenPeriodGet(PWM0_BASE, PWM_GEN_1);
  uint32_t pwmFrequency = pwmClock / (loadValue + 1);
  return pwmFrequency;
}// Function to calculate the sigmoid value


void TriggerPWMSigmoidFrequency(float* actual_freq, float target_freq){
  const uint32_t totalSteps = 400;
  float sigmoidValue=0;
  float dutyEq =0;
  uint32_t frequency, pwmClock, load, step;
  
  // Analog output y=0.0003667x−0.3667
  // Duty = 0.011111*frequency−11.111111
  //UARTprintf("\r\t\tActual Freq %d\n", (int)(*actual_freq));
  for (step = 0; step <= totalSteps; step++) {
    // Calculate the sigmoid value for this step
    sigmoidValue = sigmoid((float)step / totalSteps * SIGMOID_X0 * 2);

    // Determine direction of ramp (rising or falling)
    if (*actual_freq < target_freq){
      // Rising ramp
      frequency = *actual_freq + ((target_freq - *actual_freq) * sigmoidValue);
    }else {
      // Falling ramp
      frequency = *actual_freq - ((*actual_freq - target_freq) * sigmoidValue);
    }
    // Calculate the PWM period and set it
    pwmClock = SysCtlClockGet() /64;
    load = (pwmClock / frequency) - 1;

    PWMGenPeriodSet(PWM0_BASE, PWM_GEN_1, load);

    // Set the PWM duty cycle to 50%
    PWMPulseWidthSet(PWM0_BASE, PWM_OUT_3, load / 2);

    // Set PB7 pin configured as PWM also, but with a RC that is used to
    // simulate an analog signal
    dutyEq = 0.111111111*((0.1*(float)frequency) -100);
    PWM_SetDutyCycle(dutyEq);

    // Delay to allow the change to take effect
    SysCtlDelay(SysCtlClockGet() / (100 * totalSteps));
    GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, SLEEP_PIN);
  }

  dutyEq = 0.111111111*((0.1*(float)frequency) -100);
  PWM_SetDutyCycle(dutyEq);
  *actual_freq = frequency;
  //UARTprintf("\r\t\tActual Freq %d\n", (int)(*actual_freq));
}

void xDebaunceKey(void *ptr) {
  while(1){
    // Wait for the notification from the ISR
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    // Delay for a specified period (e.g., 1000 ms)
    vTaskDelay(pdMS_TO_TICKS(170));
    // Re-enable the PORTE interrupt
    IntEnable(INT_GPIOE);
  }  
}

void GPIOPortE_Handler(){
  // Get the interrupt status
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  uint32_t status = GPIOIntStatus(GPIO_PORTE_BASE, true);

  IntDisable(INT_GPIOE); 
  //lear the interrupt flag
  GPIOIntClear(GPIO_PORTE_BASE, status);

  //Check which pin triggered bcbcbcb2interrupt
  if (status & EC_1) {
    GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, 0);
    // Handle the falling edge on PE2
    // Your code here
    //Desacelera e acelera na direcao oposta
    vTaskNotifyGiveFromISR(xChangeDirectionHandle, &xHigherPriorityTaskWoken );
    //GPIOPinWrite(GPIO_PORTB_BASE, DIR_PIN, DIR_PIN);
  	UARTprintf("\r\t\tHandler EC_1\n");
    ChangeDirStatus = status;
  }
  if (status & EC_2) {
    // Handle the falling edge on PE3
    // Your code here
  	//UARTprintf("\r\t\t\tEC_2\n");
    GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, 0);
    vTaskNotifyGiveFromISR(xChangeDirectionHandle, &xHigherPriorityTaskWoken );
  	UARTprintf("\r\t\tHandler EC_2\n");
    //GPIOPinWrite(GPIO_PORTB_BASE, DIR_PIN, 0);
    ChangeDirStatus = status;
  }

  //BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  vTaskNotifyGiveFromISR(xDebaunceKeyHandle, &xHigherPriorityTaskWoken);
  portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void NemaInterruptionConfig(){
  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);
  
  // Step 2: Configure PB5 as an input pin
  GPIOPinTypeGPIOInput(GPIO_PORTE_BASE, EC_1);
  GPIOPinTypeGPIOInput(GPIO_PORTE_BASE, EC_2);

  // Step 3: Configure the interrupt
  // 3.1: Disable the interrupt for PB5 while configuring
  GPIOIntDisable(GPIO_PORTE_BASE, EC_1);
  GPIOIntDisable(GPIO_PORTE_BASE, EC_2);

  // 3.2: Clear any prior interrupt
  GPIOIntClear(GPIO_PORTE_BASE, EC_1);
  GPIOIntClear(GPIO_PORTE_BASE, EC_2);

  // 3.3: Configure PB5 to detect falling edges
  GPIOIntTypeSet(GPIO_PORTE_BASE, EC_1, GPIO_RISING_EDGE);
  GPIOIntTypeSet(GPIO_PORTE_BASE, EC_2, GPIO_RISING_EDGE);

  GPIOPadConfigSet(GPIO_PORTE_BASE, EC_1 | EC_2, \
                  GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);

  // 3.4: Enable the interrupt for PB5
  GPIOIntEnable(GPIO_PORTE_BASE, EC_1); // Step 3: Configure the interrupt
  GPIOIntEnable(GPIO_PORTE_BASE, EC_2); // Step 3: Configure the interrupt
  
  // Step 4: Enable the GPIO Port B interrupt in the NVICasd
  IntRegister(INT_GPIOE, GPIOPortE_Handler);
  IntEnable(INT_GPIOE);

   // Create the task to re-enable the interrupt
   xTaskCreate(xDebaunceKey, "ReEnableInterrupt", configMINIMAL_STACK_SIZE+50, \
                NULL, 15, \
                &xDebaunceKeyHandle);

  xTaskCreate(xChangeDirection, "ChangeDirection", configMINIMAL_STACK_SIZE+50, \
                NULL, 14, \
                &xChangeDirectionHandle);
  
  float min_freq = 60*KILO_HZ;
  float max_freq = 65*KILO_HZ; // <- Freq Maxima da Senoide
  float actual_freq = min_freq;

  GPIOPinWrite(GPIO_PORTB_BASE, ENABLE_PIN, 0);
  GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, 0);
  TriggerPWMSigmoidFrequency(&actual_freq, max_freq);

}

void NemaConfig(){
  

  SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
  SysCtlPeripheralEnable(SYSCTL_PERIPH_PWM0);
  GPIOPinTypePWM(GPIO_PORTB_BASE, GPIO_PIN_5 | GPIO_PIN_7);
  SysCtlPWMClockSet(SYSCTL_PWMDIV_64);
  AnalogInit();
  
  // Configure PB5 as an cbcpin
  //GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, STEP_PIN);
  GPIOPinConfigure(GPIO_PB5_M0PWM3);

  // Configure the PWM generator
  PWMGenConfigure(PWM0_BASE, PWM_GEN_1, PWM_GEN_MODE_DOWN | PWM_GEN_MODE_NO_SYNC);
  
  // Set the PWM period (the frequency of the PWM signal)
  uint32_t ui32PWMClock = SysCtlClockGet() / 64;
  uint32_t ui32Load = (ui32PWMClock / PWM_FREQUENCY) - 1;
  PWMGenPeriodSet(PWM0_BASE, PWM_GEN_1, ui32Load);

  // Set the PWM duty cycle to 50%
  uint32_t ui32PulseWidth = ui32Load / 2;
  PWMPulseWidthSet(PWM0_BASE, PWM_OUT_3, ui32PulseWidth);

  // Enable the PWM output
  PWMOutputState(PWM0_BASE, PWM_OUT_3_BIT, true);
  PWMGenEnable(PWM0_BASE, PWM_GEN_1);

  GPIOUnlockPin(GPIO_PORTB_BASE, SLEEP_PIN);
  GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, DIR_PIN);
  GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, ENABLE_PIN);
  GPIOPinTypeGPIOOutput(GPIO_PORTB_BASE, SLEEP_PIN);

  GPIOPadConfigSet(GPIO_PORTB_BASE, DIR_PIN, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD);
  GPIOPadConfigSet(GPIO_PORTB_BASE, ENABLE_PIN, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD);
  GPIOPadConfigSet(GPIO_PORTB_BASE, SLEEP_PIN, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD);
  
  GPIOPinWrite(GPIO_PORTB_BASE, DIR_PIN, DIR_PIN);
  GPIOPinWrite(GPIO_PORTB_BASE, ENABLE_PIN, 1);
  GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, 0);
  
  GreenLightConfig();
  GreenLightTurnOn();

  LinearMov_Mngr.Began      = false;
  LinearMov_Mngr.Count      = 0;
  LinearMov_Mngr.CycleCount = 0;
  LinearMov_Mngr.CycleThrshld = 5;
  
  /*
  xTaskCreate(StepLoop,
               "StepLoop",
               configMINIMAL_STACK_SIZE,
               NULL,
               tskIDLE_PRIORITY + 1,
               NULL );
  */
}

void StepLoop(void* ptr){
  
  float min_freq = 1*KILO_HZ;
  float max_freq = 10*KILO_HZ;
  float actual_freq = min_freq;

  //PWM_SetDutyCycle(10);
  while(1){
    
    
    if(actual_freq<=(min_freq+100)){
      //UARTprintf("\r\n\t\t\tRise\n\n");
      //TriggerPWMSigmoidFrequency(&actual_freq, max_freq);
      //UARTprintf("\r\n\t\t\tCurve Done\n\n");
      vTaskDelay(pdMS_TO_TICKS(1500));
    }else{
      //UARTprintf("\r\n\t\t\tFalling\n\n");
      //TriggerPWMSigmoidFrequency(&actual_freq, min_freq);
      //UARTprintf("\r\n\t\t\tCurve Done\n\n");
      vTaskDelay(pdMS_TO_TICKS(1500));
    }


    //GPIOPinWrite(GPIO_PORTB_BASE, GPIO_PIN_5, GPIOPinRead(GPIO_PORTB_BASE, GPIO_PIN_5)^GPIO_PIN_5);
    //vTaskDelay(pdMS_TO_TICKS(50));
  }
}
void NemaEnable(){
  GPIOPinWrite(GPIO_PORTB_BASE, ENABLE_PIN, 0);
}

void NemaDisable(){
  GPIOPinWrite(GPIO_PORTB_BASE, ENABLE_PIN, ENABLE_PIN);
}

void LinearMovValidation(){

  // DEBUG GPIO CONFIG
  GPIOUnlockPin(LINEAR_MOV_DBG_BASE, LINEAR_MOV_DBG);
  GPIOPinTypeGPIOOutput(LINEAR_MOV_DBG_BASE, LINEAR_MOV_DBG);
  GPIOPinWrite(LINEAR_MOV_DBG_BASE, LINEAR_MOV_DBG, 0);
  
  //Timer Configuration
  SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0);
    
  // Wait for the Timer 0 module to be ready
  while(!SysCtlPeripheralReady(SYSCTL_PERIPH_TIMER0)) {}

  // Configure Timer 0 as a 32-bit periodic timer
  TimerConfigure(TIMER0_BASE, TIMER_CFG_PERIODIC);

  // Get the system clock frequency
  uint32_t ui32SysClock  = SysCtlClockGet()/2;

  // Nao entendi pq colocando x2 funciona 
  uint32_t micro_sec_res = SysCtlClockGet()/(1000000*2);
  uint32_t mili_sec_res  = SysCtlClockGet()/(1000*2);

  uint32_t freqHez = 800;
  // Load Timer 0 for 5 seconds
  //TimerLoadSet(TIMER0_BASE, TIMER_A, ui32SysClock *5);
    TimerLoadSet(TIMER0_BASE, TIMER_A, 640*mili_sec_res*2);
  UARTprintf("\rSystem Freq %u\n", ui32SysClock);


  //CallBack
  TimerIntRegister(TIMER0_BASE, TIMER_A, Timer0IntHandler);
  TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
  IntMasterEnable();
  TimerEnable(TIMER0_BASE, TIMER_A);
  

}

void xChangeDirection(void *ptr){

  uint32_t status = GPIOIntStatus(GPIO_PORTE_BASE, true);
  uint32_t Prvstatus = 10;
  float min_freq = 0.3*KILO_HZ;
  float max_freq = 0.8*KILO_HZ;
  float actual_freq = min_freq;
  
  while(1){

    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    status = GPIOIntStatus(GPIO_PORTE_BASE, true);
  	//UARTprintf("\r\t\tGPIO Status %x\n", status);
    if (ChangeDirStatus & EC_1 && Prvstatus!=ChangeDirStatus) {
      Prvstatus=ChangeDirStatus;
  	  UARTprintf("\r\t\t\t\tEC_1\n");
        //TriggerPWMSigmoidFrequency(&actual_freq, min_freq);
      actual_freq = min_freq;
      GPIOPinWrite(GPIO_PORTB_BASE, DIR_PIN, 0);
      GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, 0);
      //TriggerPWMSigmoidFrequency(&actual_freq, max_freq);
      if(!LinearMov_Mngr.Began){
  	    UARTprintf("\r\t\t\t\tLinear 0 Began\n");
        LinearMov_Mngr.Began=true;
        LinearMov_Mngr.LimitSwitch_id = 0;
        LinearMov_Mngr.Count++;
      }else if(LinearMov_Mngr.LimitSwitch_id==0)
        LinearMov_Mngr.Count++;
    }
    if (ChangeDirStatus & EC_2 && Prvstatus!=ChangeDirStatus) {
      Prvstatus=ChangeDirStatus;
  	  UARTprintf("\r\t\tEC_2\n");
      //TriggerPWMSigmoidFrequency(&actual_freq, min_freq);
      GPIOPinWrite(GPIO_PORTB_BASE, DIR_PIN, DIR_PIN);
      GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, 0);
      actual_freq = min_freq;
      //TriggerPWMSigmoidFrequency(&actual_freq, max_freq);
      if(!LinearMov_Mngr.Began){
  	    UARTprintf("\r\t\t\t\tLinear 1 Began\n");
        LinearMov_Mngr.Began=true;
        LinearMov_Mngr.LimitSwitch_id = 1;
        LinearMov_Mngr.Count++;
      }else if(LinearMov_Mngr.LimitSwitch_id==1)
        LinearMov_Mngr.Count++;
    }
    if(LinearMov_Mngr.Count==2){
      LinearMov_Mngr.Count = 0;
      LinearMov_Mngr.CycleCount++;
  	  UARTprintf("\r\t\t\t\tLinear CycleCount %d\n",LinearMov_Mngr.CycleCount);
    }
    if(LinearMov_Mngr.CycleCount>=LinearMov_Mngr.CycleThrshld){
      GreenLightTurnOff();
      UARTprintf("\r\t\t\t\t\tNEMA Disable\n");
      GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, SLEEP_PIN);
      NemaDisable();
    }
    
    vTaskDelay(pdMS_TO_TICKS(1)/4);
    //SysCtlDelay(1*SysCtlClockGet()/1000000);
    GPIOPinWrite(GPIO_PORTB_BASE, SLEEP_PIN, SLEEP_PIN);
  }
}

#endif
