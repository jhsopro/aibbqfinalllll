

#include <SPI.h>
#include "max6675.h"

// 腳位（Arduino UNO）
const int PIN_SO  = 12;
const int PIN_CS  = 9;
const int PIN_SCK = 13;

MAX6675 thermocouple(PIN_SCK, PIN_CS, PIN_SO);

void setup() {
  Serial.begin(115200);
  delay(1000);  

  Serial.println("=== MAX6675 TEST START ===");
}

void loop() {
  double tempC = thermocouple.readCelsius();

  if (isnan(tempC)) {
    Serial.println("ERROR: NO THERMOCOUPLE");
  } else {
    Serial.print("TEMP_C:");
    Serial.println(tempC, 2);
  }

  delay(2000);  
}
