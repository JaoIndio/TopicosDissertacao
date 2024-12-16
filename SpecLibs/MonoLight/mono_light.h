#ifndef MONO_LIGHT_H
#define MONO_LIGHT_H

#define GREEN_LIGHT_PIN       GPIO_PIN_4
#define GREEN_LIGHT_PORT      GPIO_PORTB_BASE

bool GreenLightConfig();
bool GreenLightTurnOn();
bool GreenLightTurnOff();

#endif // AS7341_H


