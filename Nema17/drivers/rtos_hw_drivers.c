/*
 * rtos_hw_drivers
 *
 * Copyright (C) 2022 Texas Instruments Incorporated
 * 
 * 
 *  Redistribution and use in source and binary forms, with or without 
 *  modification, are permitted provided that the following conditions 
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the 
 *    documentation and/or other materials provided with the   
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
*/

#include <stdbool.h>
#include <stdint.h>
#include "inc/hw_gpio.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "driverlib/gpio.h"
#include "driverlib/pin_map.h"
#include "driverlib/rom.h"
#include "driverlib/rom_map.h"
#include "driverlib/sysctl.h"
#include "drivers/rtos_hw_drivers.h"

//*****************************************************************************
//
// Holds the current, debounced state of each button.  A 0 in a bit indicates
// that that button is currently pressed, otherwise it is released.
// We assume that we start with all the buttons released (though if one is
// pressed when the application starts, this will be detected).
//
//*****************************************************************************
static uint8_t g_ui8ButtonStates = ALL_BUTTONS;

//*****************************************************************************
//
//! Configures the device pins for the standard usages on the EK-TM4C1294XL.
//!
//! \param bEthernet is a boolean used to determine function of Ethernet pins.
//! If true Ethernet pins are  configured as Ethernet LEDs.  If false GPIO are
//! available for application use.
//! \param bUSB is a boolean used to determine function of USB pins. If true USB
//! pins are configured for USB use.  If false then USB pins are available for
//! application use as GPIO.
//!
//! This function enables the GPIO modules and configures the device pins for
//! the default, standard usages on the EK-TM4C1294XL.  Applications that
//! require alternate configurations of the device pins can either not call
//! this function and take full responsibility for configuring all the device
//! pins, or can reconfigure the required device pins after calling this
//! function.
//!
//! \return None.
//
//*****************************************************************************
void
PinoutSet(bool bUSB)
{
    //
    // Enable all the GPIO peripherals.
    //
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOB);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOC);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOD);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOE);
    MAP_SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);

    //
    // PA0-1 are used for UART0.
    //
    MAP_GPIOPinConfigure(GPIO_PA0_U0RX);
    MAP_GPIOPinConfigure(GPIO_PA1_U0TX);
    MAP_GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);

    //
    // PD4/PD5 are used for USB for device mode. For host mode, consult
	// spna243 for how to configure the EK-TM4C123GXL LaunchPad to operate
	// in USB host mode.
    //
    if(bUSB)
    {
        MAP_GPIOPinTypeUSBAnalog(GPIO_PORTD_BASE, GPIO_PIN_4 | GPIO_PIN_5);
    }

    //
    // Unlock PF0 so we can change it to a GPIO input.
    //
    MAP_GPIOUnlockPin(GPIO_PORTF_BASE, GPIO_PIN_0);

    //
    // PF0 and PF4 are used for user buttons.
    //
    MAP_GPIOPinTypeGPIOInput(GPIO_PORTF_BASE, GPIO_PIN_0 | GPIO_PIN_4);
    MAP_GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_0 | GPIO_PIN_4, 0);

    //
    // PF1, PF2, and PF3 are used for controlling the RGB LED.
    //
    MAP_GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3);
    MAP_GPIOPadConfigSet(GPIO_PORTF_BASE, GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3,
                             GPIO_STRENGTH_12MA, GPIO_PIN_TYPE_STD);

    //
    // Default the LEDs to OFF.
    //
    MAP_GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3, 0);
}

//*****************************************************************************
//
//! This function writes a state to the LED bank.
//!
//! \param ui32LEDMask is a bit mask for which GPIO should be changed by this
//! call.
//! \param ui32LEDValue is the new value to be applied to the LEDs after the
//! ui32LEDMask is applied.
//!
//! The first parameter acts as a mask.  Only bits in the mask that are set
//! will correspond to LEDs that may change.  LEDs with a mask that is not set
//! will not change. This works the same as GPIOPinWrite. After applying the
//! mask the setting for each unmasked LED is written to the corresponding
//! LED port pin via GPIOPinWrite.
//!
//! \return None.
//
//*****************************************************************************
void
LEDWrite(uint32_t ui32LEDMask, uint32_t ui32LEDValue)
{
    //
    // Check the mask and set or clear the LED as directed.
    //
    if(ui32LEDMask & RED_LED)
    {
        if(ui32LEDValue & RED_LED)
        {
            MAP_GPIOPinWrite(LED_PORT, RED_LED_PIN, RED_LED_PIN);
        }
        else
        {
            MAP_GPIOPinWrite(LED_PORT, RED_LED_PIN, 0);
        }
    }

    if(ui32LEDMask & BLUE_LED)
    {
        if(ui32LEDValue & BLUE_LED)
        {
            MAP_GPIOPinWrite(LED_PORT, BLUE_LED_PIN, BLUE_LED_PIN);
        }
        else
        {
            MAP_GPIOPinWrite(LED_PORT, BLUE_LED_PIN, 0);
        }
    }

    if(ui32LEDMask & GREEN_LED)
    {
        if(ui32LEDValue & GREEN_LED)
        {
            MAP_GPIOPinWrite(LED_PORT, GREEN_LED_PIN, GREEN_LED_PIN);
        }
        else
        {
            MAP_GPIOPinWrite(LED_PORT, GREEN_LED_PIN, 0);
        }
    }
}

//*****************************************************************************
//
//! This function reads the state to the LED bank.
//!
//! \param pui32LEDValue is a pointer to where the LED value will be stored.
//!
//! This function reads the state of the LaunchPad LEDs and stores that state
//! information into the variable pointed to by pui32LEDValue.
//!
//! \return None.
//
//*****************************************************************************
void LEDRead(uint32_t *pui32LEDValue)
{
	int32_t i32GPIOValue;
    *pui32LEDValue = 0;

    //
    // Read the pin state and set the variable bit if needed.
    //
    i32GPIOValue = MAP_GPIOPinRead(LED_PORT, RED_LED_PIN | BLUE_LED_PIN |
								   GREEN_LED_PIN);
    
	//
	// The LED's are on PF1 through PF3, so mask the result for only PF1, PF2,
	// and PF3 and then bit shift right by one.
	//
	*pui32LEDValue = ((i32GPIOValue & 0x0E) >> 1);
}

//*****************************************************************************
//
//! Polls the current state of the buttons and determines which have changed.
//!
//! \param pui8Delta points to a character that will be written to indicate
//! which button states changed since the last time this function was called.
//! This value is derived from the debounced state of the buttons.
//! \param pui8RawState points to a location where the raw button state will
//! be stored.
//!
//! This function should be called periodically by the application to poll the
//! pushbuttons.  It determines both the current debounced state of the buttons
//! and also which buttons have changed state since the last time the function
//! was called.
//!
//! In order for button debouncing to work properly, this function should be
//! called at a regular interval, even if the state of the buttons is not
//! needed that often.
//!
//! If button debouncing is not required, the the caller can pass a pointer
//! for the \e pui8RawState parameter in order to get the raw state of the
//! buttons.  The value returned in \e pui8RawState will be a bit mask where
//! a 1 indicates the buttons is pressed.
//!
//! \return Returns the current debounced state of the buttons where a 1 in the
//! button ID's position indicates that the button is pressed and a 0
//! indicates that it is released.
//
//*****************************************************************************
uint8_t
ButtonsPoll(uint8_t *pui8Delta, uint8_t *pui8RawState)
{
    uint32_t ui32Delta;
    uint32_t ui32Data;
    static uint8_t ui8SwitchClockA = 0;
    static uint8_t ui8SwitchClockB = 0;

    //
    // Read the raw state of the push buttons.  Save the raw state
    // (inverting the bit sense) if the caller supplied storage for the
    // raw value.
    //
    ui32Data = (MAP_GPIOPinRead(BUTTONS_GPIO_BASE, ALL_BUTTONS));
    if(pui8RawState)
    {
        *pui8RawState = (uint8_t)~ui32Data;
    }

    //
    // Determine the switches that are at a different state than the debounced
    // state.
    //
    ui32Delta = ui32Data ^ g_ui8ButtonStates;

    //
    // Increment the clocks by one.
    //
    ui8SwitchClockA ^= ui8SwitchClockB;
    ui8SwitchClockB = ~ui8SwitchClockB; 

    //
    // Reset the clocks corresponding to switches that have not changed state.
    //
    ui8SwitchClockA &= ui32Delta;
    ui8SwitchClockB &= ui32Delta;

    //
    // Get the new debounced switch state.
    //
    g_ui8ButtonStates &= ui8SwitchClockA | ui8SwitchClockB;
    g_ui8ButtonStates |= (~(ui8SwitchClockA | ui8SwitchClockB)) & ui32Data;

    //
    // Determine the switches that just changed debounced state.
    //
    ui32Delta ^= (ui8SwitchClockA | ui8SwitchClockB);

    //
    // Store the bit mask for the buttons that have changed for return to
    // caller.
    //
    if(pui8Delta)
    {
        *pui8Delta = (uint8_t)ui32Delta;
    }

    //
    // Return the debounced buttons states to the caller.  Invert the bit
    // sense so that a '1' indicates the button is pressed, which is a
    // sensible way to interpret the return value.
    //
    return(~g_ui8ButtonStates);
}

//*****************************************************************************
//
//! Initializes the GPIO pins used by the board pushbuttons.
//!
//! This function must be called during application initialization to
//! configure the GPIO pins to which the pushbuttons are attached.  It enables
//! the port used by the buttons and configures each button GPIO as an input
//! with a weak pull-up.
//!
//! \return None.
//
//*****************************************************************************
void
ButtonsInit(void)
{
    //
    // Enable the GPIO port to which the pushbuttons are connected.
    //
    MAP_SysCtlPeripheralEnable(BUTTONS_GPIO_PERIPH);

    //
    // Unlock PF0 so we can change it to a GPIO input
    // Once we have enabled (unlocked) the commit register then re-lock it
    // to prevent further changes.  PF0 is muxed with NMI thus a special case.
    //
	GPIOUnlockPin(BUTTONS_GPIO_BASE, RIGHT_BUTTON);

    //
    // Set each of the button GPIO pins as an input with a pull-up.
    //
    MAP_GPIODirModeSet(BUTTONS_GPIO_BASE, ALL_BUTTONS, GPIO_DIR_MODE_IN);
    MAP_GPIOPadConfigSet(BUTTONS_GPIO_BASE, ALL_BUTTONS,
                         GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);

    //
    // Initialize the debounced button state with the current state read from
    // the GPIO bank.
    //
    g_ui8ButtonStates = MAP_GPIOPinRead(BUTTONS_GPIO_BASE, ALL_BUTTONS);
}
