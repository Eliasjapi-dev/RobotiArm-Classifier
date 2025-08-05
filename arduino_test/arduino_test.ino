// arduino_test.ino

#include <Servo.h>

// Definir los objetos Servo para cada uno de los 6 servos
Servo servo1, servo2, servo3, servo4, servo5, servo6;

// Asignar los pines a cada servo
const int pinServo1 = 3;
const int pinServo2 = 5;
const int pinServo3 = 6;
const int pinServo4 = 9;
const int pinServo5 = 10;
const int pinServo6 = 11;

void setup() {
  // Iniciar la comunicación serial a 9600 bps
  Serial.begin(9600);
  
  // Adjuntar cada servo a su pin correspondiente
  servo1.attach(pinServo1);
  servo2.attach(pinServo2);
  servo3.attach(pinServo3);
  servo4.attach(pinServo4);
  servo5.attach(pinServo5);
  servo6.attach(pinServo6);
  
  // Posición inicial (centro) para todos los servos
  servo1.write(180);
  servo2.write(165);
  servo3.write(90);
  servo4.write(90);
  servo5.write(90);
  servo6.write(130);
  
  // Confirmar inicio
  Serial.println("Arduino listo para recibir comandos.");
}

void loop() {
  // Verificar si hay datos disponibles en el buffer serial
  if (Serial.available() > 0) {
    // Leer la línea completa hasta el carácter de nueva línea
    String command = Serial.readStringUntil('\n');
    command.trim(); // Eliminar espacios en blanco adicionales
    
    // Comandos esperados:
    // "MOVE t1 t2 t3 t4 t5 t6"
    // "GRIP OPEN"
    // "GRIP CLOSE"
    
    if (command.startsWith("MOVE")) {
      // Dividir el comando por espacios
      int t1, t2, t3, t4, t5, t6;
      sscanf(command.c_str(), "MOVE %d %d %d %d %d %d", &t1, &t2, &t3, &t4, &t5, &t6);
      
      // Limitar los ángulos entre 0 y 180 grados
      t1 = constrain(t1, 0, 180);
      t2 = constrain(t2, 0, 180);
      t3 = constrain(t3, 0, 180);
      t4 = constrain(t4, 0, 180);
      t5 = constrain(t5, 0, 180);
      t6 = constrain(t6, 0, 180);
      
      // Mover cada servo a su ángulo correspondiente
      servo1.write(t1);
      servo2.write(t2);
      servo3.write(t3);
      servo4.write(t4);
      servo5.write(t5);
      servo6.write(t6);
      
      // Confirmar movimiento
      Serial.println("MOVIMIENTO COMPLETADO");
    }
    else if (command == "GRIP OPEN") {
      // Abrir gripper (servo6 a 180 grados, por ejemplo)
      servo6.write(180);
      Serial.println("GRIPPER ABIERTOS");
    }
    else if (command == "GRIP CLOSE") {
      // Cerrar gripper (servo6 a 0 grados, por ejemplo)
      servo6.write(130);
      Serial.println("GRIPPER CERRADOS");
    }
    else {
      // Comando no reconocido
      Serial.println("COMANDO NO RECONOCIDO");
    }
  }
}
