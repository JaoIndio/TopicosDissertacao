#ifndef _DRV_LIB_H_
#define _DRV_LIB_H_

#include "LinearMov/LinMov.h"
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
#define PWM_INT GPIO_PIN_1
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
extern TaskHandle_t xDebaunceKeyHandle;
extern TaskHandle_t xChangeDirectionHandle;
extern TaskHandle_t xTrackInterfMovHandle;
extern LinMovCycle_t LinearMov_Mngr;
extern LinMovCycle_t StepCount;

void NemaConfig();
void NemaEnable();
void NemaDisable();
void NemaInterruptionConfig();
void LinearMovValidation();
void StepLoop(void* ptr);

#endif

