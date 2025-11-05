// A -> Back Left
// B -> Back Right
// C -> Front Right
// D -> Front Left

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

void moveForward(int moveTime);
void moveBackward(int moveTime);
void moveSideLeft(int moveTime);
void moveSideRight(int moveTime);
void moveRotate(int moveTime);
void testMapping();
void offAllMotor();

void setup() {
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
}

void loop() {
    //set motor A and motor B speed, 0-255 255 being the fastest
    analogWrite(PWMA,255);
    analogWrite(PWMB,255); //B is half speed
    analogWrite(PWMC,255);
    analogWrite(PWMD,255); //B is half speed

    moveRotate(5000);

    delay(2000);

    // moveSideRight(5000);

    // delay(2000);

    // for(;;);

    // testMapping();
    // delay(1000); //the two motors will stop spinning for 1 second
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

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
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

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
}

// not done
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

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
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

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
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

    delay(moveTime);

    //enable standby to make the motors stop spinning
    digitalWrite(STBY1,LOW);
    digitalWrite(STBY2,LOW);
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