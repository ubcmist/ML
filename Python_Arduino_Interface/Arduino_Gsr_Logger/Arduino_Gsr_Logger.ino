/*
This script is inspired by:
http://wiki.seeedstudio.com/Grove-GSR_Sensor/?fbclid=IwAR1cEmrWYtzrxy754C-CzfmWFNjJNa7jJbRDZ3pge6wFFXEIBvHsTq7seuk
*/

const int GSR=A0;
int sensorValue=0;
int gsr_average=0;

void setup(){
  Serial.begin(9600);
}

void loop(){
  long sum=0;
  for(int i=0;i<50;i++)           //Average the 50 measurements to remove the glitch
      {
      sensorValue=analogRead(GSR);
      sum += sensorValue;
      delay(5);
      }
   gsr_average = sum/50;
   Serial.println(gsr_average);
}
