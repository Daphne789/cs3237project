// A -> Back Left
// B -> Back Right
// C -> Front Right
// D -> Front Left

#include <WiFi.h>
#include <HTTPClient.h>

//motor A connected between A01 and A02
//motor B connected between B01 and B02
int STBY1 = 26; //standby
int STBY2 = 27; //standby
//Motor A
int PWMA = 5;  //Speed control
int AIN1 = 17; //Direction
int AIN2 = 16; //Direction
//Motor B
int PWMB = 14; 
int BIN1 = 13; 
int BIN2 = 12; 
//Motor C
int PWMC = 21;
int CIN1 = 18;
int CIN2 = 19;
//Motor D
int PWMD = 15;
int DIN1 = 0;
int DIN2 = 2;

String currentCommand = "0";
int motorSpeed = 255;

#define FORWARD 1
#define BACKWARD 2
#define LEFT 3
#define RIGHT 4
#define TIME 1000

void moveForward(int moveTime);
void moveBackward(int moveTime);
void moveSideLeft(int moveTime);
void moveSideRight(int moveTime);
void moveRotate(int moveTime);
void testMapping();
void offAllMotor();

// http
const char* ssid = "";
const char* password = "";
const char* serverName = "";

void setup() {
    Serial.begin(115200);

    pinMode(STBY1, OUTPUT);
    pinMode(STBY2, OUTPUT);
    pinMode(PWMA, OUTPUT);
    pinMode(AIN1, OUTPUT);
    pinMode(AIN2, OUTPUT);
    pinMode(PWMB, OUTPUT);
    pinMode(BIN1, OUTPUT);
    pinMode(BIN2, OUTPUT);
    pinMode(PWMC, OUTPUT);
    pinMode(CIN1, OUTPUT);
    pinMode(CIN2, OUTPUT);
    pinMode(PWMD, OUTPUT);
    pinMode(DIN1, OUTPUT);
    pinMode(DIN2, OUTPUT);

    // WiFi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
  
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println();
    Serial.println("Connected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    //set motor A and motor B speed, 0-255 255 being the fastest
    analogWrite(PWMA, motorSpeed);
    analogWrite(PWMB, motorSpeed);
    analogWrite(PWMC, motorSpeed);
    analogWrite(PWMD, motorSpeed);
}

void loop() {
    // HTTP
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;

        http.begin(serverName);
        int httpResponseCode = http.GET();
        
        if (httpResponseCode > 0) {
            Serial.print("HTTP Response code: ");
            Serial.println(httpResponseCode);
            
            // Get the response payload
            String command = http.getString();
            if (command != currentCommand) {
                currentCommand = command;
                Serial.print("New command: ");
                Serial.println(currentCommand);
            }
        }
        else {
            Serial.print("Error code: ");
            Serial.println(httpResponseCode);
            offAllMotor();
            digitalWrite(STBY1,LOW);
            digitalWrite(STBY2,LOW);
        }

        http.end();
    } else {
        Serial.println("WiFi Disconnected");
    }
    // testMapping();

    executeCommand(currentCommand);
    delay(1000);
}

void executeCommand(String command) {
    if (command == "1") {
        moveForward(TIME);
    }
    else if (command == "2") {
        moveBackward(TIME);
    }
    else if (command == "3") {
        moveSideLeft(TIME);
    }
    else if (command == "4") {
        moveSideRight(TIME);
    }
    else if (command == "0" || command == "") {
        offAllMotor();
    }
    else {
        Serial.println("Unknown command");
    }
}

void moveForward(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,HIGH);
    digitalWrite(AIN2,LOW);
    digitalWrite(BIN1,HIGH);
    digitalWrite(BIN2,LOW);
    digitalWrite(CIN1,HIGH);
    digitalWrite(CIN2,LOW);
    digitalWrite(DIN1,HIGH);
    digitalWrite(DIN2,LOW);

    // delay(moveTime);

    // //enable standby to make the motors stop spinning
    // digitalWrite(STBY1,LOW);
    // digitalWrite(STBY2,LOW);
}

void moveBackward(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,LOW);
    digitalWrite(AIN2,HIGH);
    digitalWrite(BIN1,LOW);
    digitalWrite(BIN2,HIGH);
    digitalWrite(CIN1,LOW);
    digitalWrite(CIN2,HIGH);
    digitalWrite(DIN1,LOW);
    digitalWrite(DIN2,HIGH);

    // delay(moveTime);

    // //enable standby to make the motors stop spinning
    // digitalWrite(STBY1,LOW);
    // digitalWrite(STBY2,LOW);
}


void moveSideLeft(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,HIGH);
    digitalWrite(AIN2,LOW);
    digitalWrite(BIN1,LOW);
    digitalWrite(BIN2,HIGH);
    digitalWrite(CIN1,HIGH);
    digitalWrite(CIN2,LOW);
    digitalWrite(DIN1,LOW);
    digitalWrite(DIN2,HIGH);

    // delay(moveTime);

    // //enable standby to make the motors stop spinning
    // digitalWrite(STBY1,LOW);
    // digitalWrite(STBY2,LOW);
}

void moveSideRight(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,LOW);
    digitalWrite(AIN2,HIGH);
    digitalWrite(BIN1,HIGH);
    digitalWrite(BIN2,LOW);
    digitalWrite(CIN1,LOW);
    digitalWrite(CIN2,HIGH);
    digitalWrite(DIN1,HIGH);
    digitalWrite(DIN2,LOW);

    // delay(moveTime);

    // //enable standby to make the motors stop spinning
    // digitalWrite(STBY1,LOW);
    // digitalWrite(STBY2,LOW);
}

void moveRotate(int moveTime) {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);

    digitalWrite(AIN1,LOW);
    digitalWrite(AIN2,HIGH);
    digitalWrite(BIN1,HIGH);
    digitalWrite(BIN2,LOW);
    digitalWrite(CIN1,HIGH);
    digitalWrite(CIN2,LOW);
    digitalWrite(DIN1,LOW);
    digitalWrite(DIN2,HIGH);

    // delay(moveTime);

    // //enable standby to make the motors stop spinning
    // digitalWrite(STBY1,LOW);
    // digitalWrite(STBY2,LOW);
}

void offAllMotor() {
    digitalWrite(AIN1,LOW);
    digitalWrite(AIN2,LOW);
    digitalWrite(BIN1,LOW);
    digitalWrite(BIN2,LOW);
    digitalWrite(CIN1,LOW);
    digitalWrite(CIN2,LOW);
    digitalWrite(DIN1,LOW);
    digitalWrite(DIN2,LOW);
}

void testMapping() {
    //disable standby to make the motors run
    digitalWrite(STBY1,HIGH);
    digitalWrite(STBY2,HIGH);
    
    offAllMotor();
    delay(500);
    digitalWrite(AIN1,HIGH);
    digitalWrite(AIN2,LOW);
    delay(1000);

    offAllMotor();
    delay(500);
    digitalWrite(BIN1,HIGH);
    digitalWrite(BIN2,LOW);
    delay(1000);

    offAllMotor();
    delay(500);
    digitalWrite(CIN1,HIGH);
    digitalWrite(CIN2,LOW);
    delay(1000);

    offAllMotor();
    delay(500);
    digitalWrite(DIN1,HIGH);
    digitalWrite(DIN2,LOW);
    delay(1000);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}